import uvicorn

from .api import create_app
from .config import Settings


def main():
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1, proxy_headers=True)


if __name__ == "__main__":
    main()
