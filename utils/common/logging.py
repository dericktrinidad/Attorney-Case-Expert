import logging, sys
import phoenix as px

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def setup_phoenix(project_name: str = "CinemaRAG"):
    # Launch local Phoenix UI in the background
    px.launch_app()  
    # Create a tracer for your project
    return px.Tracer(project_name=project_name)
