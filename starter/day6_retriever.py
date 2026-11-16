"""Adapters for the Day 6 PostgreSQL/pgvector index."""
import importlib.util
import json
import os
from pathlib import Path


DAY6_ROOT = Path(__file__).resolve().parents[1] / (
    'K5-L3B-DAY06-NguyenManhCuong-2A202603904-DataIngestion'
)


def load_day6_embedder_class():
    module_path = DAY6_ROOT / 'src' / 'embeddings.py'
    if not module_path.is_file():
        raise FileNotFoundError(f'Day 6 embedding adapter not found: {module_path}')
    spec = importlib.util.spec_from_file_location('day6_embeddings', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load Day 6 embedding adapter: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyEmbedder


class Day6PgVectorRetriever:
    """Search the real Day 6 pgvector index with its matching embedder."""

    def __init__(self, database_url=None, strategy='B', table_name='document_chunks', embedder=None):
        database_url = database_url or os.getenv('DATABASE_URL')
        if not database_url:
            raise RuntimeError('DATABASE_URL is required to connect to the Day 6 pgvector store')
        if not strategy:
            raise ValueError('Day 6 strategy must be a non-empty string')
        if not table_name.isidentifier():
            raise ValueError('table_name must be a SQL identifier')

        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError('Install the Day 6 dependencies, including psycopg[binary]') from exc
        try:
            from pgvector import Vector
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError('Install the Day 6 dependencies, including pgvector') from exc

        if embedder is None:
            embedder = load_day6_embedder_class()()
        if not callable(getattr(embedder, 'embed_query', None)):
            raise TypeError('embedder must provide embed_query(question)')

        self.connection = psycopg.connect(database_url, connect_timeout=3)
        register_vector(self.connection)
        self.vector_type = Vector
        self.strategy = strategy
        self.table_name = table_name
        self.embedder = embedder

    def load_chunks(self):
        """Load the selected index's text and provenance for lexical retrieval."""
        sql = (
            f'SELECT chunk_id, chunk_text, document_id, source, page, metadata '
            f'FROM {self.table_name} WHERE strategy = %s ORDER BY chunk_id'
        )
        with self.connection.cursor() as cursor:
            cursor.execute(sql, (self.strategy,))
            rows = cursor.fetchall()
        return [
            {
                'id': row[0],
                'text': row[1],
                'source_id': row[2],
                'source': row[3],
                'page': row[4],
                'metadata': row[5] or {},
            }
            for row in rows
        ]

    def search(self, query, *, filters=None, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string')
        if top_k < 1:
            raise ValueError('top_k must be positive')
        vector = self.vector_type(self.embedder.embed_query(query))
        clauses = ['strategy = %s', 'embedding IS NOT NULL']
        params = [self.strategy]
        if filters:
            for key, value in filters.items():
                clauses.append('metadata @> %s::jsonb')
                params.append(json.dumps({key: value}, ensure_ascii=False))
        sql = (
            f'SELECT chunk_id, chunk_text, document_id, source, page, metadata, '
            f'1 - (embedding <=> %s) AS score FROM {self.table_name} '
            f'WHERE {" AND ".join(clauses)} ORDER BY embedding <=> %s LIMIT %s'
        )
        params = [vector, *params, vector, top_k]
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [
            {
                'id': row[0],
                'text': row[1],
                'source_id': row[2],
                'source': row[3],
                'page': row[4],
                'metadata': row[5] or {},
                'score': float(row[6]),
                'strategy': 'dense_pgvector',
            }
            for row in rows
        ]

    def close(self):
        self.connection.close()
