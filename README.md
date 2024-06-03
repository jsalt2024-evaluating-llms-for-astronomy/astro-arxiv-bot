# AstroArXivBot
A question-answering bot powered by retrieval-augmented generation (RAG).

# Installation

## Requirements

You can install the requirements (tested on Linux Ubuntua 22.04) by pip installing from the `./requirements.txt` file. It is recommended to do this in a virtual environment (e.g., from miniconda).

In order to run the LLM with retrieval, you'll want to install the following:
```
chromadb
huggingface-hub
llama-index
openai
pydantic
transformers
torch
```

If you want to deploy to Slack, then you'll also want to install `slack-bolt` and `slack_sdk`.

## Configuring the Slack bot

You can create a new Slack app by clicking on the green "Create New App" on [this page](https://api.slack.com/apps). 

**Important**: When you install your bot to a Slack workspace, you will need to navigate to "Basic Information" under the Settings left-hand menu, and create an App-level token with the scope `connections:write`. This app token will begin with `xapp-`. Afterwards, you should navigate to "OAuth & Permissions" under Features on the left-hand menu, and copy the OAuth Token that begins with `xoxb-`. *Store these in a secret `.env` file within this directory, and ensure these do not get shared anywhere public and/or are not saved in your git history.*

You will need to configure your app so that it can listen in to all the relevant events and react appropriately. We can quickly do this via the app manifest:

```json
{
    "display_information": {
        "name": "<name of your Slack bot>",
        "description": "<Some description of your Salck bot>",
        "background_color": "<your favorite hex color>"
    },
    "features": {
        "app_home": {
            "home_tab_enabled": false,
            "messages_tab_enabled": true,
            "messages_tab_read_only_enabled": false
        },
        "bot_user": {
            "display_name": "<display name of your Slack bot>",
            "always_online": true
        }
    },
    "oauth_config": {
        "scopes": {
            "bot": [
                "app_mentions:read",
                "channels:history",
                "chat:write",
                "emoji:read",
                "groups:history",
                "im:history",
                "im:read",
                "im:write",
                "links:read",
                "links:write",
                "mpim:history",
                "mpim:read",
                "reactions:read",
                "reactions:write",
                "channels:read"
            ]
        }
    },
    "settings": {
        "event_subscriptions": {
            "bot_events": [
                "app_mention",
                "message.channels",
                "message.groups",
                "message.im",
                "message.mpim",
                "reaction_added",
                "reaction_removed"
            ]
        },
        "interactivity": {
            "is_enabled": true
        },
        "org_deploy_enabled": false,
        "socket_mode_enabled": true,
        "token_rotation_enabled": false
    }
}
```

If you don't require all these features, then you may want to cull the list of OAuth permissions (scopes) or the Event subscriptions (bot events).


## Running the bot

Once you have everything configured, you will need a server to host the bot. You can run the bot by executing, e.g., 

```sh
python src/llm-chatbot.py --local-llm -k 5 -v DEBUG
```

in order to run a local model (LLaMA-3-8B-Instruct) that retrieves the top 5 papers, and has verbose logs (DEBUG mode).

# Citation

If you use this software, then please consider citing the accompanying paper:

```
@misc{astroarxivbot-llm,
      title={Designing an Evaluation Framework for Large Language Models in Astronomy Research}, 
      author={John F. Wu and Alina Hyk and Kiera McCormick and Christine Ye and Simone Astarita and Elina Baral and Jo Ciuca and Jesse Cranney and Anjalie Field and Kartheik Iyer and Philipp Koehn and Jenn Kotler and Sandor Kruk and Michelle Ntampaka and Charles O'Neill and Joshua E. G. Peek and Sanjib Sharma and Mikaeel Yunus},
      year={2024},
      eprint={2405.20389},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
}
```
