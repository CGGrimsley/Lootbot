"""Discord bot for inventory detection using YOLO and PaddleOCR."""

import asyncio
import datetime
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import aiofiles
import cv2
import discord
import numpy as np
import torch
from discord.ext import commands, tasks
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageOps
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants
YOLO_MODEL_PATH = Path("best.pt")
TEMP_DIR = Path("temp")
DEBUG_DIR = Path("debug_crops")
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory to store each user's inventory data
INVENTORY_DATA_DIR = Path("inventory_data")
INVENTORY_DATA_DIR.mkdir(exist_ok=True)

# File to store the current wipe info
CURRENT_WIPE_FILE = INVENTORY_DATA_DIR / "current_wipe.json"

# For each wipe a folder is created under “wipes” with two subfolders: “regular” and “farm”
WIPE_DATA_DIR = INVENTORY_DATA_DIR / "wipes"
WIPE_DATA_DIR.mkdir(exist_ok=True)


# ---------------------------
# Current Wipe Info Helpers
# ---------------------------
def load_current_wipe() -> Optional[Dict]:
    if CURRENT_WIPE_FILE.exists():
        try:
            with CURRENT_WIPE_FILE.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading current wipe: {e}")
    return None


def save_current_wipe(wipe_info: Dict) -> None:
    with CURRENT_WIPE_FILE.open("w") as f:
        json.dump(wipe_info, f, indent=2)


# Admin username for commands
ADMIN_USERNAME = "admin_user_name"

# Item classification config
ITEM_CATEGORIES = {
    "Sulfur_stack": "Sulfur",
    "gunpowder": "Gunpowder",
    "explosives": "Explosives",
    "cooked_sulfur": "Cooked Sulfur",
    "pipes": "Pipes",
    "AK47": "AK47",
    "Metal_ore": "Metal Ore",
    "Diesel": "Diesel",
    "High_quality_metal": "High-Quality Metal",
    "Crude_oil": "Crude Oil",
    "Cloth": "Cloth",
    "Scrap": "Scrap",
    "HQM_ore": "HQM Ore",
    "Rocket": "Rocket",
    "c4": "C4",
    "charcoal": "Charcoal",
    "MLRS": "MLRS",
    "MLRS_module": "MLRS Module",
    "Metal_fragments": "Metal Fragments",
    "Low_grade_fuel": "Low Grade Fuel",
}

# Maximum detection limits
MAX_QUANTITIES_PER_DETECTION = {
    "Diesel": 40,
    "Rocket": 6,
    "AK47": 1,
    "Charcoal": 8000,
    "Gunpowder": 2000,
    # Default max limit for other items
    "default": 4000,
}


def cap_detection_quantity(class_name: str, quantity: int) -> int:
    """Cap the quantity for a single detection."""
    max_limit = MAX_QUANTITIES_PER_DETECTION.get(
        class_name, MAX_QUANTITIES_PER_DETECTION["default"]
    )
    if quantity > max_limit:
        logger.warning(
            f"Quantity for {class_name} detection ({quantity}) exceeds max limit ({max_limit}), "
            f"capping to {max_limit}."
        )
    return min(quantity, max_limit)


def get_user_file(user_id: int) -> Path:
    """Returns the Path to the JSON file storing this user's data."""
    return INVENTORY_DATA_DIR / f"user_{user_id}.json"


def load_user_data(user_id: int) -> Dict:
    """
    Loads the user's inventory data from a JSON file.
    If not found, returns a default structure.
    """
    file_path = get_user_file(user_id)
    if file_path.exists():
        try:
            with file_path.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse JSON for user {user_id}: {e}")
            # Fallback to new data if file is corrupted
            return {"inventory": {}, "last_image": {}}
    else:
        return {"inventory": {}, "last_image": {}}


def save_user_data(user_id: int, data: Dict) -> None:
    """
    Saves the user's inventory data to a JSON file.
    """
    file_path = get_user_file(user_id)
    try:
        with file_path.open("w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save user data for user {user_id}: {e}")


def add_new_detections_to_inventory(
    user_id: int,
    new_detections: Dict[str, int],
    wipe_id: str,
    inv_type: str = "regular",
) -> None:
    """
    Loads the user's data from disk (for a specific wipe and inventory type),
    applies the new detections to their total inventory, updates 'last_image' with these detections,
    and records a timestamped update.
    """
    # Use the wipe-specific version of load_user_data
    data = load_user_data(user_id, wipe_id, inv_type)

    # Update total inventory
    for item_name, quantity in new_detections.items():
        data["inventory"][item_name] = data["inventory"].get(item_name, 0) + quantity

    # Overwrite the last_image with the new detections
    data["last_image"] = new_detections

    # Record this update with a timestamp (optional, if you are using updates)
    update_record = {
        "timestamp": datetime.datetime.now().timestamp(),
        "items": new_detections,
        "type": inv_type,
    }
    data.setdefault("updates", []).append(update_record)

    # Save the updated data back to disk
    save_user_data(user_id, wipe_id, data, inv_type)


def get_user_file(user_id: int, wipe_id: str, inv_type: str = "regular") -> Path:
    """
    Returns the file path for the given user's inventory for a specific wipe and inventory type.
    """
    base = WIPE_DATA_DIR / wipe_id / inv_type
    base.mkdir(parents=True, exist_ok=True)
    return base / f"user_{user_id}.json"


def load_user_data(user_id: int, wipe_id: str, inv_type: str = "regular") -> Dict:
    """
    Loads the user's inventory data from the wipe-specific JSON file.
    """
    file_path = get_user_file(user_id, wipe_id, inv_type)
    if file_path.exists():
        try:
            with file_path.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(
                f"Failed to parse JSON for user {user_id} in wipe {wipe_id}: {e}"
            )
    return {"inventory": {}, "last_image": {}, "updates": []}


def save_user_data(
    user_id: int, wipe_id: str, data: Dict, inv_type: str = "regular"
) -> None:
    """
    Saves the user's inventory data for a given wipe and inventory type.
    """
    file_path = get_user_file(user_id, wipe_id, inv_type)
    try:
        with file_path.open("w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(
            f"Failed to save user data for user {user_id} in wipe {wipe_id}: {e}"
        )


def aggregate_inventory(wipe_id: str, inv_type: str = "regular") -> Dict[str, int]:
    """
    Aggregates and sums up inventories for all users in a given wipe.
    """
    total = {}
    folder = WIPE_DATA_DIR / wipe_id / inv_type
    if folder.exists():
        for file in folder.glob("user_*.json"):
            try:
                with file.open("r") as f:
                    data = json.load(f)
                inv = data.get("inventory", {})
                for item, qty in inv.items():
                    total[item] = total.get(item, 0) + qty
            except Exception as e:
                logger.error(f"Error aggregating file {file}: {e}")
    return total


def aggregate_yearly_inventory(year: str, inv_type: str = "regular") -> Dict[str, int]:
    """
    Aggregates inventories for all wipes within a given year.
    """
    total = {}
    if WIPE_DATA_DIR.exists():
        for wipe_folder in WIPE_DATA_DIR.iterdir():
            if wipe_folder.is_dir() and wipe_folder.name.startswith(year):
                inv = aggregate_inventory(wipe_folder.name, inv_type)
                for item, qty in inv.items():
                    total[item] = total.get(item, 0) + qty
    return total


class ImageProcessor:
    """High-accuracy image processing with PaddleOCR integration."""

    def __init__(self):
        """Initialize YOLO model and PaddleOCR."""
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.yolo_model.to(GPU_DEVICE)
        self.reader = PaddleOCR(use_angle_cls=True, lang="en")
        logger.info(f"Initialized models on {GPU_DEVICE}")

    @staticmethod
    def adaptive_preprocessing(image: Image.Image) -> list:
        """Apply a series of preprocessing steps to improve OCR accuracy."""
        processed_images = []

        # Convert to grayscale
        gray_image = image.convert("L")

        # Base enhancements (contrast, sharpness)
        enhancer = ImageEnhance.Contrast(gray_image)
        high_contrast = enhancer.enhance(2.5)

        enhancer = ImageEnhance.Sharpness(high_contrast)
        sharp_image = enhancer.enhance(2.0)

        # Inversion (optional for dark backgrounds)
        inverted = ImageOps.invert(sharp_image)

        # Resize for small text
        resized = inverted.resize((int(inverted.width * 2), int(inverted.height * 2)))

        # Local histogram equalization (via OpenCV)
        img_np = np.array(resized)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_img = clahe.apply(img_np)

        # Thresholding
        _, thresh_img = cv2.threshold(
            eq_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Add all variants for OCR attempts
        processed_images.append(Image.fromarray(eq_img))
        processed_images.append(Image.fromarray(thresh_img))
        processed_images.append(resized)

        return processed_images

    @staticmethod
    def extract_quantity(text: str) -> Optional[int]:
        """Extracts quantities from OCR text more accurately."""
        text = text.lower()

        # Fix common OCR misreads
        text = text.replace("o", "0").replace("l", "1").replace("i", "1")

        # Remove unwanted characters
        text = re.sub(r"[^0-9x]", "", text)

        # Fix misplaced 'x' (e.g., 'x10' should be '10')
        text = re.sub(r"(?<!\d)x", "", text)  # Remove 'x' if not preceded by a digit

        # Extract numbers
        numbers = re.findall(r"\d+", text)

        # Convert to integers
        valid_numbers = [int(num) for num in numbers if 1 <= int(num) < 100000]

        return max(valid_numbers) if valid_numbers else None

    def perform_ocr(self, image: Image.Image) -> Tuple[Optional[int], str]:
        """Try multiple OCR attempts and return the most confident result."""
        preprocessed_images = self.adaptive_preprocessing(image)

        best_text = ""
        best_quantity = None

        for processed_image in preprocessed_images:
            img_np = np.array(processed_image)
            results = self.reader.ocr(img_np, cls=True)

            detected_text = " ".join(
                [res[1][0] for res in results[0] if res[1][1] > 0.2]
            ).strip()

            quantity = self.extract_quantity(detected_text)

            # If OCR fails, retry with another preprocessing method
            if quantity is None and len(detected_text) < 3:
                continue  # Skip and try next

            # Pick the best number
            if quantity is not None and (
                best_quantity is None or quantity > best_quantity
            ):
                best_quantity = quantity
                best_text = detected_text

        # If all preprocessing methods fail, return 0 instead of None
        return best_quantity if best_quantity is not None else 0, best_text

    def process_image(self, image_path: Path) -> Dict[str, int]:
        """Processes an inventory image using YOLO object detection and OCR."""
        results = self.yolo_model(image_path)
        inventory = {}
        fixed_size = 100  # Adjust for your image resolution

        for detection in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection
            class_name = results[0].names[int(class_id)]

            if class_name not in ITEM_CATEGORIES:
                logger.debug("Skipping unknown class: %s", class_name)
                continue

            with Image.open(image_path) as img:
                width, height = img.size
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                crop_box = (
                    max(0, center_x - fixed_size / 2),
                    max(0, center_y - fixed_size / 2),
                    min(width, center_x + fixed_size / 2),
                    min(height, center_y + fixed_size / 2),
                )
                cropped = img.crop(crop_box)

                debug_path = DEBUG_DIR / f"{class_name}_{int(x1)}_{int(y1)}.png"
                cropped.save(debug_path)
                logger.info("Saved crop for %s at %s", class_name, debug_path)

                quantity, text = self.perform_ocr(cropped)

                # Retry OCR for HQM Ore with more aggressive approach
                if class_name == "HQM_ore" and quantity == 0:
                    logger.warning(
                        "Retrying OCR for HQM Ore with enhanced preprocessing..."
                    )
                    for preprocessed in self.adaptive_preprocessing(cropped):
                        quantity, text = self.perform_ocr(preprocessed)
                        if quantity:
                            break

                if quantity is None:
                    logger.warning(
                        "OCR failed for %s, defaulting quantity to 0 or 1.", class_name
                    )
                    quantity = 1 if class_name == "AK47" else 0

                # Cap quantity
                quantity = cap_detection_quantity(class_name, quantity)

            item_name = ITEM_CATEGORIES[class_name]
            inventory[item_name] = inventory.get(item_name, 0) + quantity

        return inventory


def clean_debug_folder():
    """Deletes all files inside the `debug_crops` folder before restart."""
    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    DEBUG_DIR.mkdir(exist_ok=True)
    logger.info("Cleared debug_crops folder.")


class CVBot(commands.Bot):
    """Discord bot with PaddleOCR and YOLO-based inventory detection."""

    def __init__(self, *args, **kwargs):
        """Initialize the bot with an image processor."""
        super().__init__(*args, **kwargs)
        self.image_processor = ImageProcessor()

    async def on_ready(self):
        """Called when the bot is fully connected."""
        logger.info("Bot ready as %s", self.user)
        if not self.restart_loop.is_running():
            self.restart_loop.start()

    @tasks.loop(hours=1)
    async def restart_loop(self):
        """
        Restarts the bot every hour by exiting the script.
        An external script (like run_bot.py) must detect closure and restart.
        """
        logger.info(
            "Hourly scheduled restart triggered. Cleaning debug folder, then exiting."
        )
        clean_debug_folder()
        await self.close()
        sys.exit(0)  # Let an external process or script restart us

    async def process_attachment(
        self, attachment: discord.Attachment, user_id: int
    ) -> str:
        """
        Handle image attachments and process inventory detection.
        Returns:
            str: Newly detected items/quantities as a string.
        """
        TEMP_DIR.mkdir(exist_ok=True)
        file_path = TEMP_DIR / attachment.filename

        try:
            await attachment.save(file_path, use_cached=True)
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)

            # Detect new inventory from the image.
            new_detections = self.image_processor.process_image(temp_path)

            # Load the current wipe info; if not set, return an error message.
            current_wipe = load_current_wipe()
            if not current_wipe:
                return "Current wipe not set."

            # Update the user's inventory using the wipe-specific helper.
            add_new_detections_to_inventory(
                user_id, new_detections, current_wipe["id"], "regular"
            )

            # Return only newly detected items.
            return (
                "\n".join(f"{k}: {v}" for k, v in new_detections.items())
                or "No items detected."
            )

        except Exception as e:
            logger.error("Processing error: %s", e, exc_info=True)
            return f"Error processing image: {str(e)}"
        finally:
            # Cleanup temporary files.
            for path in [file_path, temp_path]:
                if path.exists():
                    try:
                        path.unlink(missing_ok=True)
                    except Exception as exc:
                        logger.warning("Cleanup error: %s", exc)


# Bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = CVBot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    """Run when the bot is online."""
    logger.info("Bot ready as %s", bot.user)


@bot.event
async def on_message(message: discord.Message):
    """Handle messages with attachments in 'loot-brags' channel."""
    if message.author == bot.user:
        return

    # If the message is in a specific channel and has attachments
    if message.channel.name == "loot-brags" and message.attachments:
        for attachment in message.attachments:
            result = await bot.process_attachment(attachment, message.author.id)
            await message.channel.send(f"Detected:\n{result}")

    # Process other bot commands (e.g., !inventory, !append, etc.)
    await bot.process_commands(message)


@bot.command()
async def inventory(ctx: commands.Context):
    """
    Displays the user's total inventory for the current wipe.
    Usage: !inventory
    """
    # First, load the current wipe information.
    current_wipe = load_current_wipe()
    if not current_wipe:
        await ctx.send("Current wipe not set.")
        return

    # Use the wipe ID from the current wipe info.
    wipe_id = current_wipe["id"]
    user_id = ctx.author.id

    # Now call the updated load_user_data that requires a wipe_id.
    data = load_user_data(user_id, wipe_id, "regular")
    user_inv = data.get("inventory", {})

    if not user_inv:
        await ctx.send("You have no recorded inventory yet.")
        return

    results = "\n".join(f"{item}: {count}" for item, count in user_inv.items())
    await ctx.send(f"Your current inventory:\n{results}")


@bot.command()
async def append(ctx: commands.Context):
    """
    Allows the user to correct the items from their last detected image.
    Usage: !append
    """
    user_id = ctx.author.id
    data = load_user_data(user_id)
    last_image = data.get("last_image", {})

    if not last_image:
        await ctx.send("No previous detections found to modify.")
        return

    # Show items from the last image
    items_list = "\n".join(
        f"{idx+1}. {item}: {count}"
        for idx, (item, count) in enumerate(last_image.items())
    )
    await ctx.send(f"Select an item to modify:\n{items_list}")

    def check_item_choice(msg):
        return (
            msg.author == ctx.author
            and msg.content.isdigit()
            and 1 <= int(msg.content) <= len(last_image)
        )

    try:
        response = await bot.wait_for("message", check=check_item_choice, timeout=30)
        selected_idx = int(response.content) - 1
        selected_item = list(last_image.keys())[selected_idx]
        old_quantity = last_image[selected_item]

        # Ask for new quantity
        await ctx.send(
            f"Enter the new quantity for {selected_item} (old = {old_quantity}):"
        )

        def check_quantity(msg):
            return msg.author == ctx.author and msg.content.isdigit()

        response = await bot.wait_for("message", check=check_quantity, timeout=30)
        new_quantity = int(response.content)

        # Update total inventory by diff
        diff = new_quantity - old_quantity
        data["inventory"][selected_item] = (
            data["inventory"].get(selected_item, 0) + diff
        )

        # Update last_image
        data["last_image"][selected_item] = new_quantity

        # If zero, remove from last_image
        if new_quantity == 0:
            data["last_image"].pop(selected_item, None)

        # If total is zero or negative, remove from overall inventory
        if data["inventory"].get(selected_item, 0) <= 0:
            data["inventory"].pop(selected_item, None)

        save_user_data(user_id, data)

        await ctx.send(
            f"Updated {selected_item} from {old_quantity} to {new_quantity}."
        )
    except asyncio.TimeoutError:
        await ctx.send("Operation timed out.")
    except Exception as e:
        logger.error(f"Error in append command: {e}", exc_info=True)
        await ctx.send("Something went wrong. Please try again.")


@bot.command()
async def restart(ctx: commands.Context):
    """
    Admin-only command to restart the bot immediately.
    The external script (run_bot.py or similar) will detect the exit and relaunch.
    """
    if ctx.author.name != ADMIN_USERNAME:
        return await ctx.send("No permission.")

    await ctx.send("Restarting bot...")
    await bot.close()
    sys.exit(0)


@bot.command()
async def clearinv(ctx, target: str = None):
    """
    Clears inventory data for the current wipe.

    Usage:
      - !clearinv
          Clears your own inventory for the current wipe.
      - !clearinv all
          (Admin only) Clears inventory for all users in the current wipe.
      - !clearinv @user or !clearinv username
          (Admin only) Clears inventory for a specific user in the current wipe.

    This command only deletes inventory files within the current wipe folder.
    """
    # If a target is provided other than your own inventory, require admin permission.
    if target is not None and ctx.author.name != ADMIN_USERNAME:
        return await ctx.send(
            "You do not have permission to clear inventories for others."
        )

    # Load the current wipe info.
    current_wipe = load_current_wipe()
    if not current_wipe:
        return await ctx.send("Current wipe not set.")

    wipe_id = current_wipe["id"]
    # Get the folder for the current wipe's regular inventories.
    inv_folder = WIPE_DATA_DIR / wipe_id / "regular"

    # Case 1: No argument provided -> clear the invoking user's own inventory.
    if target is None:
        user_file = get_user_file(ctx.author.id, wipe_id, "regular")
        if user_file.exists():
            try:
                user_file.unlink(missing_ok=True)
                await ctx.send(
                    f"Your inventory has been cleared for the current wipe ({wipe_id})."
                )
            except Exception as e:
                logger.error(f"Error deleting file {user_file}: {e}")
                await ctx.send("An error occurred while clearing your inventory.")
        else:
            await ctx.send("You have no inventory data for the current wipe.")

    # Case 2: Target is "all" -> clear all inventories for the current wipe.
    elif target.lower() == "all":
        count = 0
        # Loop through all user inventory files in the current wipe's regular folder.
        for file in inv_folder.glob("user_*.json"):
            try:
                file.unlink(missing_ok=True)
                count += 1
            except Exception as e:
                logger.error(f"Error deleting file {file}: {e}")
        await ctx.send(
            f"Cleared inventory data for {count} user(s) in the current wipe ({wipe_id})."
        )

    # Case 3: Target is a specific user (mention or username).
    else:
        # First, try to get the member from mentions.
        member = None
        if ctx.message.mentions:
            member = ctx.message.mentions[0]
        else:
            # Try to search by username (case-insensitive).
            member = discord.utils.find(
                lambda m: m.name.lower() == target.lower(), ctx.guild.members
            )

        if not member:
            return await ctx.send("User not found.")

        user_file = get_user_file(member.id, wipe_id, "regular")
        if user_file.exists():
            try:
                user_file.unlink(missing_ok=True)
                await ctx.send(
                    f"Cleared inventory data for {member.display_name} in the current wipe ({wipe_id})."
                )
            except Exception as e:
                logger.error(f"Error deleting file {user_file}: {e}")
                await ctx.send(
                    "An error occurred while clearing that user's inventory."
                )
        else:
            await ctx.send("No inventory data found for that user in the current wipe.")


@bot.command()
async def wipe(ctx, start_date: str, end_date: str):
    """
    Admin-only: Sets the current wipe period.
    Usage: !wipe YYYY-MM-DD YYYY-MM-DD
    """
    # Check if the user is allowed to use this command.
    if ctx.author.name != ADMIN_USERNAME:
        return await ctx.send("No permission.")

    # Validate the input dates using datetime.strptime.
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        # If the date format is wrong, notify the user.
        return await ctx.send("Invalid date format. Use YYYY-MM-DD.")

    # Create a wipe ID from the dates.
    wipe_id = f"{start_date}_to_{end_date}"
    wipe_info = {
        "id": wipe_id,
        "start": start_date,
        "end": end_date,
        "year": start_date.split("-")[0],
    }

    # Save the current wipe information.
    save_current_wipe(wipe_info)

    # Create directory structure for the new wipe.
    (WIPE_DATA_DIR / wipe_id / "regular").mkdir(parents=True, exist_ok=True)
    (WIPE_DATA_DIR / wipe_id / "farm").mkdir(parents=True, exist_ok=True)

    # Respond with a success embed.
    embed = discord.Embed(
        title="Wipe Set",
        description=f"Current wipe set to {start_date} to {end_date}",
        color=0x00FF00,
    )
    await ctx.send(embed=embed)


# ---------------------------
# Error Handler for the !wipe Command
# ---------------------------
@wipe.error
async def wipe_error(ctx, error):
    """
    Handle errors for the !wipe command.
    In particular, if required arguments are missing, provide a friendly error message.
    """
    if isinstance(error, commands.MissingRequiredArgument):
        embed = discord.Embed(
            title="Error: Missing Argument",
            description="Missing required arguments.\nUsage: `!wipe YYYY-MM-DD YYYY-MM-DD`",
            color=0xFF0000,
        )
        await ctx.send(embed=embed)
    else:
        # Re-raise other errors so they can be handled by the global error handler or show a traceback.
        raise error


@bot.command()
async def farm(ctx):
    """
    Processes attachments as farm inventory.
    Usage: !farm (with an image attachment)

    The detections are stored in the current wipe’s "farm" inventory.
    """
    # Ensure a current wipe is set.
    current_wipe = load_current_wipe()
    if not current_wipe:
        return await ctx.send("Current wipe not set. Contact admin.")

    # Ensure the message contains an attachment.
    if not ctx.message.attachments:
        return await ctx.send("No attachment found.")

    results = []
    for attachment in ctx.message.attachments:
        try:
            # Process the image attachment (similar to process_attachment)
            TEMP_DIR.mkdir(exist_ok=True)
            file_path = TEMP_DIR / attachment.filename

            await attachment.save(file_path, use_cached=True)
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)

            # Detect items from the image.
            new_detections = bot.image_processor.process_image(temp_path)

            # Update the user's inventory in the "farm" subfolder.
            add_new_detections_to_inventory(
                ctx.author.id, new_detections, current_wipe["id"], "farm"
            )

            # Prepare the result summary.
            results.append(
                "\n".join(f"{k}: {v}" for k, v in new_detections.items())
                or "No items detected."
            )

        except Exception as e:
            logger.error("Processing error (farm): %s", e, exc_info=True)
            results.append(f"Error processing image: {str(e)}")
        finally:
            # Cleanup temporary files.
            for path in [file_path, temp_path]:
                if path.exists():
                    try:
                        path.unlink(missing_ok=True)
                    except Exception as exc:
                        logger.warning("Cleanup error: %s", exc)

    # Send the results in an embed.
    embed = discord.Embed(
        title="Farm Inventory Update", description="\n".join(results), color=0x00FF00
    )
    await ctx.send(embed=embed)


@bot.command()
async def total(ctx, year: str = None):
    """
    Displays the combined regular inventory totals.

    Without an argument: totals for the current wipe.
    With a year (YYYY) argument: totals aggregated from all wipes in that year.
    """
    if year is None:
        # Get current wipe info.
        current_wipe = load_current_wipe()
        if not current_wipe:
            return await ctx.send("Current wipe not set.")
        wipe_id = current_wipe["id"]
        total_inv = aggregate_inventory(wipe_id, "regular")
        title = f"Total Inventory for Wipe {wipe_id}"
    else:
        # Aggregate totals for all wipes in the given year.
        total_inv = aggregate_yearly_inventory(year, "regular")
        title = f"Total Inventory for Year {year}"

    if not total_inv:
        return await ctx.send("No inventory data found.")

    # Build the embed message.
    embed = discord.Embed(title=title, color=0x00FF00)
    for item, qty in total_inv.items():
        embed.add_field(name=item, value=str(qty), inline=True)

    await ctx.send(embed=embed)


@bot.command()
async def leaderboard(ctx):
    """
    Displays the top 5 users (by weighted points) from the current wipe’s farm inventories.

    Weights:
      - Sulfur: 2 points
      - Metal Fragments: 1 point
      - HQM Ore: 300 points
    """
    current_wipe = load_current_wipe()
    if not current_wipe:
        return await ctx.send("Current wipe not set.")

    wipe_id = current_wipe["id"]
    user_points = {}
    # Define the weights for items.
    POINTS = {"Sulfur": 2, "Metal Fragments": 1, "HQM Ore": 300}
    # Folder for farm inventories.
    farm_folder = WIPE_DATA_DIR / wipe_id / "farm"

    if not farm_folder.exists():
        return await ctx.send("No farm inventories found.")

    # Loop through each user file and compute the weighted score.
    for file in farm_folder.glob("user_*.json"):
        try:
            with file.open("r") as f:
                data = json.load(f)
            inv = data.get("inventory", {})
            total_points = 0
            for item, qty in inv.items():
                weight = POINTS.get(item, 0)
                total_points += weight * qty
            # Extract user id from the filename format "user_<id>.json"
            user_id = int(file.stem.split("_")[1])
            user_points[user_id] = total_points
        except Exception as e:
            logger.error(f"Error processing leaderboard for {file}: {e}")

    if not user_points:
        return await ctx.send("No farm inventories found.")

    # Sort users by points in descending order and take top 5.
    sorted_users = sorted(user_points.items(), key=lambda x: x[1], reverse=True)[:5]

    embed = discord.Embed(title="Farm Leaderboard", color=0xFFD700)
    for rank, (user_id, points) in enumerate(sorted_users, start=1):
        member = ctx.guild.get_member(user_id)
        name = member.display_name if member else str(user_id)
        embed.add_field(name=f"{rank}. {name}", value=f"{points} points", inline=False)

    await ctx.send(embed=embed)


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_TOKEN")
    if not TOKEN:
        raise ValueError("Missing DISCORD_TOKEN in environment")

    DEBUG_DIR.mkdir(exist_ok=True)
    bot.run(TOKEN)
