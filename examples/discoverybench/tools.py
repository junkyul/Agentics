"""Text2Sql search Toll"""

from ddgs import DDGS

from mcp.server.fastmcp import FastMCP
from agentic_db import AgenticDB
from agentics_discovery import Dataset
mcp = FastMCP("Search")

dataset=Dataset.import_from_discovery_bench_metadata("introduction_pathways_non-native_plants")
dataset.dbs[0].import_db_from_csv()



@mcp.tool()
def execute_sql(sql_query: str, max_results: int) -> dict | None:
    """execute a sql query aganinst the target db and return a dataframe in json format
    """
    dataset.dbs[0].query_db(sql_query)
    
    

if __name__ == "__main__":

    mcp.run(transport="stdio")
