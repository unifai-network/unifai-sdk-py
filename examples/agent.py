import dotenv
dotenv.load_dotenv()

import os

import unifai
from unifai.client import DiscordClient, TelegramClient, TwitterClient

if __name__ == '__main__':
    agent = unifai.Agent(api_key=os.getenv("UNIFAI_AGENT_API_KEY", ""))

    if os.getenv("TELEGRAM_BOT_TOKEN"):
        telegram_client = TelegramClient(bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""))
        agent.add_client(telegram_client)

    if os.getenv("DISCORD_BOT_TOKEN"):
        discord_client = DiscordClient(bot_token=os.getenv("DISCORD_BOT_TOKEN", ""))
        agent.add_client(discord_client)

    if os.getenv("TWITTER_API_KEY"):
        twitter_client = TwitterClient(
            api_key=os.getenv("TWITTER_API_KEY", ""),
            api_secret=os.getenv("TWITTER_API_SECRET", ""),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
            access_secret=os.getenv("TWITTER_ACCESS_SECRET", ""),
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
            bot_screen_name=os.getenv("TWITTER_BOT_SCREEN_NAME", ""),
        )
        agent.add_client(twitter_client)

    agent.run()
