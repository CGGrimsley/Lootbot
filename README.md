# Lootbot - The Rust loot bot (BETA)

This Discord bot uses [YOLO](https://github.com/ultralytics/ultralytics) for object detection and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for text recognition to identify and track user inventories from posted screenshots of the survival video game RUST. (e.g., in game loot)

### Features
- **Object Detection** with YOLO to find item icons in screenshots (>95% accuracy).
- **Text Recognition** with PaddleOCR to read item counts (>92% accuracy).
- **Persistent Inventories** stored locally in JSON, tracking each user’s detected items over time.
- **Admin Controls** to reset all inventories and perform routine maintenance tasks.
- **Commands**:
  - `!inventory` – Display the total inventory of the user.
  - `!append` – Correct the last detected item count if OCR made a mistake.
  - `!clearinv` – Admin-only command to wipe inventory data for the current wipe (for a specific user, all users or yourself).
  ```
  !clearinv @user
  !clearinv all
  !clearinv
  ```
  - `!restart` - Admin-only command to force restart of the bot.
  - `!wipe` - Admin-only command to configure dates for the current wipe (YYYY-MM-DD)
  - `!total` - Display the users current total for the current wipe, or can show yearly totals using `!total YYYY`.
  - `!farm` - Used alongside an attachment to count detections towards the leaderboard.
  - `!leaderboard` - Generates a leaderboard of the top 5 contributors for the current wipe using a weighted point system.

### Requirements
- **Python**: 3.10
- **Dependencies**: Listed in [`requirements.txt`](./requirements.txt)

### Setup & Installation
1. **Clone** this repository:
   ```bash
   git clone https://github.com/CGGrimsley/Lootbot.git
   cd Lootbot
   ```

2. Install Dependencies
   ```pip install -r requirements.txt```

3. Replace the example token in .env with your Discord bot token:
   ```
   DISCORD_TOKEN=YourDiscordBotTokenHere
   ```

4. Replace the default Admin Username in bot.py with your username:
   ```
   ADMIN_USERNAME = "admin_user_name"
   ```

6. Ensure your discord has a proper channel for the bot titled the following:
   ```loot-brags```

### Contributions
  Contributions are welcome! Please feel free to reach out me on discord @kdrgold or via github!

### License
  [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)


