import discord
from discord.ext import commands
from ultralytics import YOLO
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Tesseract Path (Modify this to your local Tesseract installation path if needed)
pytesseract.pytesseract.tesseract_cmd = r"Path\to\tesseract"
lang = "eng"

# Load the YOLO Model (Ensure 'best.pt' is in the same directory or specify the correct path)
yolo_model = YOLO("best.pt")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Item Categories Mapping
ITEM_CATEGORIES = {
    'Sulfur_stack': 'Sulfur',
    'gunpowder': 'Gunpowder',
    'explosives': 'Explosives',
    'cooked_sulfur': 'Cooked Sulfur',
    'pipes': 'Pipes',
    'AK47': 'AK47',
    'Metal_ore': 'Metal Ore',
    'Diesel': 'Diesel',
    'High_quality_metal': 'High-Quality Metal',
    'Crude_oil': 'Crude Oil',
    'Cloth': 'Cloth',
    'Scrap': 'Scrap',
    'HQM_ore': 'HQM Ore',
    'Rocket': 'Rocket',
    'c4': 'C4',
    'charcoal': 'Charcoal',
    'MLRS': 'MLRS',
    'MLRS_module': 'MLRS Module',
    'Metal_fragments': 'Metal Fragments',
    'Low_grade_fuel': 'Low Grade Fuel'
}

# Discord Bot Setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Function to preprocess cropped images for better OCR
def preprocess_image(image, method="default"):
    image = image.convert("L")  # Convert to grayscale
    if method == "default":
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image
        image = ImageOps.autocontrast(image)  # Adjust contrast automatically
        image = image.point(lambda p: 0 if p < 200 else 255, '1')  # Binarize white text on black background
    elif method == "adaptive_threshold":
        image = ImageOps.autocontrast(image)
        image = image.point(lambda p: 0 if p < 128 else 255, '1')  # Adaptive thresholding
    elif method == "edge_enhance":
        image = image.filter(ImageFilter.EDGE_ENHANCE)
    elif method == "denoise":
        image = image.filter(ImageFilter.MedianFilter(size=3))  # Denoising filter
    elif method == "contrast_boost":
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Boost contrast
    return image

# Function to sanitize filenames
def sanitize_filename(filename):
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(c if c in valid_chars else '_' for c in filename).strip()[:100]

def extract_quantity(text):
    """
    Extracts the first valid number from the OCR text. Handles commas and leading 'x'.
    """
    match = re.search(r"x?\b[\d,]+\b", text)
    if match:
        try:
            # Remove x and commas, then convert to int
            return int(match.group(0).replace("x", "").replace(",", ""))
        except ValueError:
            pass  # If conversion fails, return None
    return None

# Function to perform iterative OCR processing with explicit "x" handling
def perform_ocr(cropped_region):
    """
    Performs OCR with multiple preprocessing methods and uses LSTM OCR engine.
    """
    methods = ["default", "adaptive_threshold", "edge_enhance", "denoise", "contrast_boost"]
    for method in methods:
        preprocessed_image = preprocess_image(cropped_region, method=method)
        # Use LSTM OCR engine with PSM mode 6
        text = pytesseract.image_to_string(preprocessed_image, config="--oem 1 --psm 6").strip()
        # Log raw OCR text for debugging
        print(f"Raw OCR Text ({method}): {text}")
        # Ensure "x" is handled correctly and extract numeric values
        text = re.sub(r"[^\dx,]", "", text)  # Keep only digits, x, and commas
        quantity = extract_quantity(text)
        if quantity is not None:
            return quantity, text
    return None, "fallback"  # Return fallback text if no quantity found

# Function to process images
def process_image(image_path):
    # Perform YOLO inference
    results = yolo_model(image_path)

    # Filter results based on ITEM_CATEGORIES
    detections = [box for box in results[0].boxes.data.cpu().numpy()]
    inventory = {}

    # Loop through each detection
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_name = results[0].names[int(class_id)]
        if class_name in ITEM_CATEGORIES:
            # Crop the detected area (expand slightly to include the quantity text)
            image = Image.open(image_path)
            cropped_region = image.crop((x1, y1, x2, y2 + 30))  # Increased crop area to capture text

            # Perform iterative OCR
            quantity, text = perform_ocr(cropped_region)

            # Save the cropped region for debugging
            debug_folder = "debug_crops"
            os.makedirs(debug_folder, exist_ok=True)
            sanitized_text = sanitize_filename(text)
            debug_filename = os.path.join(debug_folder, f"{class_name}_{sanitized_text}.png")
            cropped_region.save(debug_filename)

            # Fallback for quantity
            if quantity is None:
                if class_name == 'AK47':
                    quantity = 1  # Each AK47 detection counts as 1
                else:
                    quantity = 1  # Default to 1 if no quantity is found

            # Update inventory
            item_name = ITEM_CATEGORIES[class_name]
            inventory[item_name] = inventory.get(item_name, 0) + quantity

    return inventory

# Bot Commands
@bot.event
async def on_ready():
    print(f"Bot is ready. Logged in as {bot.user}")

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the message is in a channel named "loot-brags" and contains attachments
    if message.channel.name == "loot-brags" and message.attachments:
        for attachment in message.attachments:
            # Save the image locally
            file_path = f"temp/{attachment.filename}"
            os.makedirs("temp", exist_ok=True)
            await attachment.save(file_path)

            # Process the image
            try:
                inventory = process_image(file_path)
                inventory_text = "\n".join([f"{item}: {count}" for item, count in inventory.items()])
                await message.channel.send(f"Detected inventory:\n{inventory_text}")
            except Exception as e:
                await message.channel.send(f"Error processing the image: {e}")

            # Clean up
            os.remove(file_path)

    # Ensure other bot commands still work
    await bot.process_commands(message)

@bot.command()
async def inventory(ctx):
    if not ctx.message.attachments:
        await ctx.send("Please attach an image containing the inventory.")
        return

    for attachment in ctx.message.attachments:
        # Save the image locally
        file_path = f"temp/{attachment.filename}"
        os.makedirs("temp", exist_ok=True)
        await attachment.save(file_path)

        # Process the image
        try:
            inventory = process_image(file_path)
            inventory_text = "\n".join([f"{item}: {count}" for item, count in inventory.items()])
            await ctx.send(f"Detected inventory:\n{inventory_text}")
        except Exception as e:
            await ctx.send(f"Error processing the image: {e}")

        # Clean up
        os.remove(file_path)

# Run the bot
TOKEN = os.getenv("DISCORD_TOKEN")  # Load token from .env file
if not TOKEN:
    raise ValueError("Discord bot token not found in .env file")

bot.run(TOKEN)
