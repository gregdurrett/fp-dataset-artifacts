
# Getting Started

1. Install Poetry

Poetry is a Python package and virtual environment manager.

Following the [Installation](https://python-poetry.org/docs/#installation) from Poetry website.

2. Create virtual environment and install Python dependencies.

```sh
poetry install
```

3. Add `OPENAI_API_KEY` to dotenv

```sh
echo "OPENAI_API_KEY=XXX" > .env
```

> You can get your API key at [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)

4. Start Jupyter

```sh
./jupyter.sh
```

