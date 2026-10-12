import asyncio

import httpx
import pytest

from sprout_sandbox import Sandbox
from sprout_sandbox.pool import MicroVMPool


@pytest.mark.parametrize("failure", [None, "http", "stream"])
def test_download_streams_binary_data_and_never_publishes_partial_files(tmp_path, failure):
    payload = bytes(range(256)) * 4096
    destination = tmp_path / "logs.tar.gz"
    destination.write_bytes(b"previous archive")

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield payload[:1000]
            if failure == "stream":
                raise httpx.ReadError("stream lost")
            yield payload[1000:]

    async def scenario():
        pool = MicroVMPool("http://server", api_key="test")
        await pool._client.aclose()

        def handle(request):
            assert request.method == "GET"
            assert request.url.params["path"] == "/tmp/logs.tar.gz"
            assert request.headers[pool.SANDBOX_ID_HEADER] == "vm-test"
            assert request.headers[pool.TARGET_PORT_HEADER] == pool.ENVD_PORT
            assert request.headers["X-API-Key"] == "test"
            return httpx.Response(404 if failure == "http" else 200, stream=Stream())

        pool._client = httpx.AsyncClient(transport=httpx.MockTransport(handle),
                                       headers=pool._client.headers)
        try:
            assert pool.supports_download()
            if failure:
                with pytest.raises(httpx.HTTPError):
                    await pool.download_file(Sandbox(backend=None, _container_id="vm-test"),
                                             "/tmp/logs.tar.gz", destination)
            else:
                await pool.download_file(Sandbox(backend=None, _container_id="vm-test"),
                                         "/tmp/logs.tar.gz", destination)
        finally:
            await pool._client.aclose()

    asyncio.run(scenario())
    assert destination.read_bytes() == (b"previous archive" if failure else payload)
    assert list(tmp_path.iterdir()) == [destination]
