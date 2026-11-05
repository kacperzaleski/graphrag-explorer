"""Neo4j access: one async driver per process.

The driver holds a connection pool and is expensive to create, so it is built once (in FastAPI's
lifespan handler from Phase 3 on) and shared. Sessions are cheap and short-lived: one per request.
"""

from neo4j import AsyncDriver, AsyncGraphDatabase

from backend.config import Settings


def create_driver(settings: Settings) -> AsyncDriver:
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )


async def server_info(driver: AsyncDriver) -> str:
    """Fails fast with a clear error if Neo4j is down or the password is wrong."""
    await driver.verify_connectivity()
    records, _, _ = await driver.execute_query(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name + ' ' + versions[0] + ' (' + edition + ')' AS info"
    )
    return records[0]["info"]
