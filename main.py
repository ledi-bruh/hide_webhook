import asyncio
import typing as t
import uuid
import uvicorn
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)

    api_base_url: str
    api_key: SecretStr
    api_agent_id: int = 0
    cache_timeout: int = 300
    host: str
    port: int


class WikibotApi:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key

    async def ask(
        self,
        chat_id: str,
        query: str,
        format: str,
        msg_id: str,
        attachments: list[str],
        agent_id: int,
    ) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f'{self.base_url}/ask',
                headers={'Authorization': self.api_key},
                json={
                    'chatId': chat_id,
                    'query': query,
                    'format': format,
                    'msgId': msg_id,
                    'attachments': attachments,
                    'agentId': agent_id
                }
            ) as response:
                response.raise_for_status()

    async def set_webhook(self, url: str) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f'{self.base_url}/set-webhook-url',
                headers={'Authorization': self.api_key},
                json={'url': url},
            ) as response:
                response.raise_for_status()


class InMemoryCache:
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self.data: dict[t.Hashable, asyncio.Future[dict]] = {}

    async def set(self, key: t.Hashable, result: dict) -> None:
        if f := self.data.get(key):
            f.set_result(result)

    async def get(self, key: t.Hashable) -> asyncio.Future[dict]:
        f = self.data.setdefault(key, asyncio.Future())
        f.add_done_callback(lambda f: self.data.pop(key, None))
        asyncio.get_running_loop().call_later(self.timeout, lambda: (f.cancel(), self.data.pop(key, None)))
        return f


load_dotenv()
config = Config()  # noqa

server = FastAPI()
server.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_wikibot = WikibotApi(config.api_base_url, config.api_key.get_secret_value())
_cache = InMemoryCache(timeout=config.cache_timeout)


async def get_cache() -> InMemoryCache:
    return _cache


async def get_wikibot() -> WikibotApi:
    return _wikibot


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None
    scenario_id: str | None = None


class AskResponse(BaseModel):
    status: str
    query: str
    answer: str
    session_id: str
    scenario_id: str


@server.post('/ask', response_model=AskResponse)
async def ask(
    req: AskRequest,
    cache: InMemoryCache = Depends(get_cache),
    wikibot: WikibotApi = Depends(get_wikibot),
) -> AskResponse:
    session_id =  str(uuid.uuid4()) if req.session_id is None else req.session_id
    scenario_id =  str(uuid.uuid4()) if req.scenario_id is None else req.scenario_id

    status = 'success'

    fut = await cache.get(key=session_id)

    try:
        _resp = await wikibot.ask(
            chat_id=session_id,
            query=req.query,
            format='raw',
            msg_id='',
            attachments=[],
            agent_id=config.api_agent_id,
        )
    except Exception as _exc:
        status = 'error'

    try:
        response_data = await fut
    except asyncio.CancelledError:
        status = 'timeout'
        response_data = {}

    return AskResponse(
        status=status,
        query=req.query,
        answer=response_data.get('answer', ''),
        session_id=session_id,
        scenario_id=scenario_id,
    )


@server.post('/webhook')
async def webhook(req: dict, cache: InMemoryCache = Depends(get_cache)):
    await cache.set(
        key=req.get('chatId', ''),
        result=req,
    )


if __name__ == '__main__':
    print(config)
    uvicorn.run(
        app=server,
        host=config.host,
        port=config.port,
    )
