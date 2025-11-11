"""
Database for storing metadata and clinical information
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MetadataDatabase:
    """
    SQLite database for storing image metadata and clinical information
    """
    
    def __init__(self, db_path: str):
        """
        Initialize database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        
        self._create_tables()
        logger.info(f"Initialized database at {db_path}")
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Main images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                uid TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                filename TEXT,
                projection TEXT,
                
                -- Clinical text fields
                findings TEXT,
                impression TEXT,
                indication TEXT,
                comparison TEXT,
                
                -- Structured fields (stored as JSON)
                mesh TEXT,
                problems TEXT,
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Embeddings table (optional: cache embeddings)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                uid TEXT PRIMARY KEY,
                image_embedding BLOB,
                findings_embedding BLOB,
                impression_embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (uid) REFERENCES images(uid) ON DELETE CASCADE
            )
        """)
        
        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_projection ON images(projection)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON images(created_at)
        """)
        
        self.conn.commit()
    
    def insert(self, data: Dict[str, Any]):
        """
        Insert a record
        
        Args:
            data: Dictionary with image information
        """
        cursor = self.conn.cursor()
        
        # Convert lists to JSON strings
        mesh_json = json.dumps(data.get('mesh', [])) if data.get('mesh') else None
        problems_json = json.dumps(data.get('problems', [])) if data.get('problems') else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO images 
            (uid, image_path, filename, projection, findings, impression, 
             indication, comparison, mesh, problems)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('uid'),
            data.get('image_path'),
            data.get('filename'),
            data.get('projection'),
            data.get('findings'),
            data.get('impression'),
            data.get('indication'),
            data.get('comparison'),
            mesh_json,
            problems_json
        ))
        
        self.conn.commit()
    
    def insert_many(self, data_list: List[Dict[str, Any]]):
        """
        Insert multiple records efficiently
        
        Args:
            data_list: List of dictionaries
        """
        cursor = self.conn.cursor()
        
        records = []
        for data in data_list:
            mesh_json = json.dumps(data.get('mesh', [])) if data.get('mesh') else None
            problems_json = json.dumps(data.get('problems', [])) if data.get('problems') else None
            
            records.append((
                data.get('uid'),
                data.get('image_path'),
                data.get('filename'),
                data.get('projection'),
                data.get('findings'),
                data.get('impression'),
                data.get('indication'),
                data.get('comparison'),
                mesh_json,
                problems_json
            ))
        
        cursor.executemany("""
            INSERT OR REPLACE INTO images 
            (uid, image_path, filename, projection, findings, impression, 
             indication, comparison, mesh, problems)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        self.conn.commit()
        logger.info(f"Inserted {len(records)} records")
    
    def get(self, uid: str) -> Optional[Dict]:
        """
        Get a record by UID
        
        Args:
            uid: Unique identifier
            
        Returns:
            Dictionary with record data or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM images WHERE uid = ?", (uid,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Convert row to dict and parse JSON fields
        result = dict(row)
        result['mesh'] = json.loads(result['mesh']) if result['mesh'] else []
        result['problems'] = json.loads(result['problems']) if result['problems'] else []
        
        return result
    
    def get_many(self, uids: List[str]) -> List[Dict]:
        """
        Get multiple records by UIDs
        
        Args:
            uids: List of unique identifiers
            
        Returns:
            List of dictionaries
        """
        if not uids:
            return []
        
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(uids))
        cursor.execute(f"SELECT * FROM images WHERE uid IN ({placeholders})", uids)
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['mesh'] = json.loads(result['mesh']) if result['mesh'] else []
            result['problems'] = json.loads(result['problems']) if result['problems'] else []
            results.append(result)
        
        return results
    
    def search_by_text(self, query: str, field: str = 'findings', limit: int = 100) -> List[Dict]:
        """
        Simple text search in a field
        
        Args:
            query: Search query
            field: Field to search in
            limit: Maximum number of results
            
        Returns:
            List of matching records
        """
        cursor = self.conn.cursor()
        
        # Simple LIKE search (can be improved with FTS)
        cursor.execute(f"""
            SELECT * FROM images 
            WHERE {field} LIKE ? 
            LIMIT ?
        """, (f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['mesh'] = json.loads(result['mesh']) if result['mesh'] else []
            result['problems'] = json.loads(result['problems']) if result['problems'] else []
            results.append(result)
        
        return results
    
    def get_all(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all records
        
        Args:
            limit: Optional limit on number of records
            
        Returns:
            List of all records
        """
        cursor = self.conn.cursor()
        
        if limit:
            cursor.execute("SELECT * FROM images LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM images")
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['mesh'] = json.loads(result['mesh']) if result['mesh'] else []
            result['problems'] = json.loads(result['problems']) if result['problems'] else []
            results.append(result)
        
        return results
    
    def count(self) -> int:
        """Get total number of records"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images")
        return cursor.fetchone()[0]
    
    def delete(self, uid: str):
        """Delete a record"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM images WHERE uid = ?", (uid,))
        self.conn.commit()
    
    def cache_embedding(self, uid: str, embedding_type: str, embedding: Any):
        """
        Cache an embedding
        
        Args:
            uid: Unique identifier
            embedding_type: Type of embedding (image, findings, impression)
            embedding: Embedding data (will be serialized)
        """
        import pickle
        
        cursor = self.conn.cursor()
        
        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)
        
        # Check if record exists
        cursor.execute("SELECT uid FROM embeddings WHERE uid = ?", (uid,))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute(f"""
                UPDATE embeddings 
                SET {embedding_type}_embedding = ?
                WHERE uid = ?
            """, (embedding_blob, uid))
        else:
            cursor.execute(f"""
                INSERT INTO embeddings (uid, {embedding_type}_embedding)
                VALUES (?, ?)
            """, (uid, embedding_blob))
        
        self.conn.commit()
    
    def get_cached_embedding(self, uid: str, embedding_type: str) -> Optional[Any]:
        """
        Get cached embedding
        
        Args:
            uid: Unique identifier
            embedding_type: Type of embedding
            
        Returns:
            Embedding or None if not cached
        """
        import pickle
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT {embedding_type}_embedding FROM embeddings WHERE uid = ?
        """, (uid,))
        
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        
        # Deserialize
        return pickle.loads(row[0])
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {
            'total_images': self.count(),
        }
        
        # Count by projection
        cursor.execute("""
            SELECT projection, COUNT(*) as count 
            FROM images 
            GROUP BY projection
        """)
        stats['by_projection'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count with MeSH
        cursor.execute("""
            SELECT COUNT(*) FROM images WHERE mesh IS NOT NULL AND mesh != '[]'
        """)
        stats['with_mesh'] = cursor.fetchone()[0]
        
        # Count with problems
        cursor.execute("""
            SELECT COUNT(*) FROM images WHERE problems IS NOT NULL AND problems != '[]'
        """)
        stats['with_problems'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")


def test_database():
    """Test the database"""
    print("=== Testing Metadata Database ===\n")
    
    # Create database
    db = MetadataDatabase("test_metadata.db")
    
    # Insert sample data
    sample_data = {
        'uid': 'test_001',
        'image_path': '/path/to/image.jpg',
        'filename': 'image.jpg',
        'projection': 'AP',
        'findings': 'Normal chest X-ray',
        'impression': 'No acute findings',
        'mesh': ['Lung', 'Thorax'],
        'problems': ['screening']
    }
    
    db.insert(sample_data)
    print("Inserted sample data")
    
    # Retrieve data
    retrieved = db.get('test_001')
    print(f"\nRetrieved data: {retrieved}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics: {stats}")
    
    # Clean up
    db.delete('test_001')
    db.close()
    
    print("\n✓ Database test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_database()
