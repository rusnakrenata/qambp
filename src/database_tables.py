import os

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Load variables from a local .env file if python-dotenv is installed.
# The repository ships a .env.example template; copy it to .env and fill it in.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass

# Connection settings are read from the environment so that no credentials
# are committed to the repository.
db_user = os.getenv("DB_USER", "user")
db_password = os.getenv("DB_PASSWORD", "password")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "3306")
db_name = os.getenv("DB_NAME", "graphPartitioning")

# DATABASE_URL takes precedence when set, which allows pointing the code at
# a throwaway backend (e.g. sqlite:///test.db) without a MariaDB server.
connection_url = os.getenv("DATABASE_URL") or (
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

# Create SQLAlchemy engine
engine = create_engine(connection_url)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to MariaDB successful!")
except Exception as e:
    print(f"Error connecting to MariaDB: {e}")


DESCRIPTION_CALIBRATION = "No description"   # lambda_mult sweep, hybrid solver
DESCRIPTION_EVALUATION = "regression_KLPM"   # GBR-predicted lambda, hybrid solver
DESCRIPTION_QPU = "qpu"                      # direct-QPU runs

Base = declarative_base()
# Graphs Table
class Graph(Base):
    __tablename__ = 'graphs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    # NOTE: MySQL/MariaDB requires an explicit VARCHAR length; a bare String
    # makes Base.metadata.create_all() fail on a fresh database.
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    nr_of_nodes = Column(Integer) 
    edge_probability = Column(Float)
    edges = Column(JSON)  # Store list of nodes as JSON
    pymetis_partition_nodes = Column(JSON, nullable=True)
    pymetis_inter_edges = Column(Integer, nullable=True)
    pymetis_intra_edges = Column(Integer, nullable=True)
    lambda_est = Column(Float, nullable=True)
    kernighan_lin_inter_edges = Column(Integer, nullable=True)
    density = Column(Float, nullable=True)
    clustering = Column(Float, nullable=True)
    nr_of_edges=Column(Integer, nullable=True)
    duration_pymetis = Column(Float, nullable=True)
    duration_keringham_lin = Column(Float, nullable=True)

    
    # Relationship to Partition table
    partitions = relationship("Partition", back_populates="graph", cascade="all, delete-orphan")


# Partitions Table
class Partition(Base):
    __tablename__ = 'partitions'
    # Removed autoincrement from 'id' as composite key
    id = Column(Integer, primary_key=True)  # Keep it a normal column, no autoincrement
    graph_id = Column(Integer, ForeignKey('graphs.id'), primary_key=True)  # ForeignKey referring to 'graphs.id'
    comp_type = Column(Text, nullable=True)
    nodes = Column(Text, nullable=True)  # Store list of nodes as JSON
    qa_inter_edges = Column(Integer, nullable=True)
    lambda_mult = Column(Float, nullable=True)
    lambda_total = Column(Float, nullable=True)
    duration_qa = Column(Float, nullable=True)
    duration_full = Column(Float, nullable=True) # qa + matrix calc

    # Relationship back to Graph
    graph = relationship("Graph", back_populates="partitions")
