import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from day6_retriever import Day6PgVectorRetriever


class Day6RetrieverTests(unittest.TestCase):
    def test_missing_database_does_not_fall_back_to_fake_dense(self):
        with self.assertRaisesRegex(RuntimeError, 'DATABASE_URL is required'):
            Day6PgVectorRetriever(database_url='')


if __name__ == '__main__':
    unittest.main()
