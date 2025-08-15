# RAG Chat with LlamaIndex and Heroku AI

This is a RAG chat application that uses [LlamaIndex](https://www.llamaindex.ai/) and [Heroku AI](https://www.heroku.com/ai) to answer questions about a set of documents. It uses [Streamlit](https://www.streamlit.io/) for the UI and Heroku AI for the LLM and vector database.

> [!NOTE]
> This is a fork of the [Streamlit on Heroku](https://github.com/heroku-reference-apps/heroku-streamlit) reference app.

## Features

- RAG chat with LlamaIndex
- Heroku AI for the LLM
- Vector database with pgvector for semantic search

## Quick start - Installation instructions

The fastest & easiest way to get started is to choose option 1 below: automatic deployment on Heroku.

### 1. Heroku - Automatic Deployment (faster & easier)

First, click on this handy dandy button:
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://www.heroku.com/deploy?template=https://github.com/heroku-reference-apps/mia-llamaindex-rag)

It will take a couple of minutes for your app to deploy, and then you'll be able to click links to 1) manage your app, and 2) view your live, interactive Streamlit app featuring RAG chat with LlamaIndex and Heroku AI.

### 2. Heroku - Manual Deployment (does the same thing as 1, but nice for learning / understanding)

Push this repository to your app or fork this repository on github and link your
repository to your heroku app.

To create a new app and deploy the code:

``` bash
export APP_NAME=<your_app_name>

# clone this repo
git clone git@github.com:heroku-reference-apps/mia-llamaindex-rag.git
cd mia-llamaindex-rag

# Create a new app (or use an existing one you've made)
heroku create $APP_NAME

# Specify the buildpack it should use:
heroku buildpacks:add heroku/python -a $APP_NAME

# Connect your app to the repo
heroku git:remote -a $APP_NAME

# deploy
git push heroku main

# Follow the URL from ^^ to view your app! To view logs, run `heroku logs --tail -a $APP_NAME`
```

Then provision the database with:

``` bash
heroku addons:create heroku-postgresql:essential-0 -a $APP_NAME --wait
heroku pg:psql -a $APP_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Then provision the AI models with:

``` bash
heroku ai:models:create heroku-inference:claude-4-sonnet --app $APP_NAME --as INFERENCE
heroku ai:models:create heroku-inference:cohere-embed-multilingual --app $APP_NAME --as EMBEDDING
```

## Running the app locally

``` bash
# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# set environment variables
heroku config -a $APP_NAME --shell > .env

# run the app
streamlit run mia_llamaindex.py
```

## Contributing

If you want to contribute to this project, please see the [CONTRIBUTING.md](CONTRIBUTING.md) guide for more details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.
