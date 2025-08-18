//! # TYL Graph Port
//!
//! Graph database port for the TYL (The YAML Life) framework. This port defines
//! the contract for graph database operations following hexagonal architecture.
//!
//! ## Features
//!
//! - **Graph Storage**: CRUD operations for nodes and relationships
//! - **Graph Traversal**: Path finding, neighbor discovery, and graph exploration
//! - **Graph Analytics**: Centrality calculations, community detection, pattern recognition
//! - **Query Execution**: Custom query support for database-specific operations
//! - **Health Monitoring**: Connection health and statistics
//! - **Comprehensive Error Handling**: Uses TYL error framework
//! - **Mock Implementation**: Built-in mock for testing and development
//! - **Async Support**: Full async/await support throughout
//!
//! ## Architecture
//!
//! This module follows hexagonal architecture:
//!
//! - **Ports (Interfaces)**: Graph traits define the contract
//! - **Adapters**: Database-specific implementations
//! - **Domain Logic**: Graph operations independent of infrastructure
//!
//! ## Quick Start
//!
//! ```rust
//! use tyl_graph_port::{GraphStore, GraphNode, TraversalParams};
//! use tyl_errors::TylResult;
//!
//! #[tokio::main]
//! async fn main() -> TylResult<()> {
//!     // Implementation would be provided by an adapter
//!     // let store = SomeGraphAdapter::new();
//!     
//!     // Create a node
//!     // let node = GraphNode::new()
//!     //     .with_label("Person")
//!     //     .with_property("name", "Alice");
//!     // let node_id = store.create_node(node).await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for complete usage examples.

// Re-export TYL framework dependencies
pub use tyl_db_core::{
    log_database_error, DatabaseLifecycle, DatabaseLogger, DatabaseResult, DatabaseTracer,
    Environment, HealthCheckResult, HealthStatus, LogLevel, LogRecord, Logger, TraceConfig,
    TracingManager,
};
pub use tyl_errors::{TylError, TylResult};

// Graph-specific error helpers
pub mod graph_errors {
    use super::*;

    /// Create an error for when a graph node is not found
    pub fn node_not_found(id: impl Into<String>) -> TylError {
        TylError::not_found("graph_node", id.into())
    }

    /// Create an error for when a graph relationship is not found
    pub fn relationship_not_found(id: impl Into<String>) -> TylError {
        TylError::not_found("graph_relationship", id.into())
    }

    /// Create an error for invalid graph queries
    pub fn invalid_query(msg: impl Into<String>) -> TylError {
        TylError::validation("graph_query", msg.into())
    }

    /// Create an error for graph traversal issues
    pub fn traversal_error(msg: impl Into<String>) -> TylError {
        TylError::database(format!("Graph traversal error: {}", msg.into()))
    }

    /// Create an error for graph analytics issues
    pub fn analytics_error(msg: impl Into<String>) -> TylError {
        TylError::database(format!("Graph analytics error: {}", msg.into()))
    }
}

// Use DatabaseResult for consistency with TYL framework
pub type GraphResult<T> = DatabaseResult<T>;

// Core modules
mod mock;
mod types;

// Re-export main types and traits
pub use types::*;

#[cfg(feature = "mock")]
pub use mock::MockGraphStore;

use async_trait::async_trait;
use std::collections::HashMap;

/// Core graph storage operations
///
/// This trait defines the essential operations for storing and managing
/// graph data. Implementations should handle transaction management,
/// constraint validation, and performance optimization internally.
///
/// This trait extends DatabaseLifecycle to provide standard database
/// management operations alongside graph-specific functionality.
#[async_trait]
pub trait GraphStore: DatabaseLifecycle + Send + Sync {
    /// Create a new node in the graph
    ///
    /// # Arguments
    /// * `node` - Node data with labels and properties
    ///
    /// # Returns
    /// The ID of the created node
    ///
    /// # Implementation Notes
    /// - Should validate node against schema if defined
    /// - Should handle duplicate prevention if needed
    /// - Should set created_at and updated_at timestamps
    async fn create_node(&self, node: GraphNode) -> TylResult<String>;

    /// Create multiple nodes in a batch for efficiency
    ///
    /// # Arguments
    /// * `nodes` - Batch of nodes to create
    ///
    /// # Returns
    /// Vector of results (node IDs or errors)
    ///
    /// # Implementation Notes
    /// - Should be transactional where possible
    /// - Should be more efficient than multiple single creates
    async fn create_nodes_batch(
        &self,
        nodes: Vec<GraphNode>,
    ) -> TylResult<Vec<Result<String, TylError>>>;

    /// Retrieve a node by its ID
    ///
    /// # Arguments
    /// * `id` - Node ID to retrieve
    ///
    /// # Returns
    /// * `Ok(Some(node))` if found
    /// * `Ok(None)` if not found
    /// * `Err(...)` if error occurred
    async fn get_node(&self, id: &str) -> TylResult<Option<GraphNode>>;

    /// Update an existing node's properties
    ///
    /// # Arguments
    /// * `id` - Node ID to update
    /// * `properties` - New properties (will be merged with existing)
    ///
    /// # Implementation Notes
    /// - Should merge properties, not replace entirely
    /// - Should update the updated_at timestamp
    /// - Should validate against schema if defined
    async fn update_node(
        &self,
        id: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<()>;

    /// Delete a node and all its relationships
    ///
    /// # Arguments
    /// * `id` - Node ID to delete
    ///
    /// # Implementation Notes
    /// - Should cascade delete all relationships
    /// - Should be idempotent (deleting non-existent node should not error)
    async fn delete_node(&self, id: &str) -> TylResult<()>;

    /// Create a relationship between two nodes
    ///
    /// # Arguments
    /// * `relationship` - Relationship data
    ///
    /// # Returns
    /// The ID of the created relationship
    ///
    /// # Implementation Notes
    /// - Should validate that both nodes exist
    /// - Should prevent duplicate relationships if needed
    /// - Should set created_at and updated_at timestamps
    async fn create_relationship(&self, relationship: GraphRelationship) -> TylResult<String>;

    /// Get a relationship by its ID
    ///
    /// # Arguments
    /// * `id` - Relationship ID to retrieve
    ///
    /// # Returns
    /// * `Ok(Some(relationship))` if found
    /// * `Ok(None)` if not found
    /// * `Err(...)` if error occurred
    async fn get_relationship(&self, id: &str) -> TylResult<Option<GraphRelationship>>;

    /// Update a relationship's properties
    ///
    /// # Arguments
    /// * `id` - Relationship ID to update
    /// * `properties` - New properties (will be merged with existing)
    ///
    /// # Implementation Notes
    /// - Should merge properties, not replace entirely
    /// - Should update the updated_at timestamp
    async fn update_relationship(
        &self,
        id: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<()>;

    /// Delete a relationship
    ///
    /// # Arguments
    /// * `id` - Relationship ID to delete
    ///
    /// # Implementation Notes
    /// - Should be idempotent (deleting non-existent relationship should not error)
    async fn delete_relationship(&self, id: &str) -> TylResult<()>;
}

/// Graph traversal and relationship analysis
///
/// This trait provides methods for exploring graph structure,
/// finding paths, and analyzing relationships between nodes.
#[async_trait]
pub trait GraphTraversal: Send + Sync {
    /// Find all nodes directly connected to a given node
    ///
    /// # Arguments
    /// * `node_id` - Starting node ID
    /// * `params` - Traversal parameters
    ///
    /// # Returns
    /// Connected nodes with their connecting relationships
    async fn get_neighbors(
        &self,
        node_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<(GraphNode, GraphRelationship)>>;

    /// Find shortest path between two nodes
    ///
    /// # Arguments
    /// * `from_id` - Starting node ID
    /// * `to_id` - Target node ID
    /// * `params` - Traversal parameters
    ///
    /// # Returns
    /// * `Ok(Some(path))` if path exists
    /// * `Ok(None)` if no path found
    /// * `Err(...)` if error occurred
    async fn find_shortest_path(
        &self,
        from_id: &str,
        to_id: &str,
        params: TraversalParams,
    ) -> TylResult<Option<GraphPath>>;

    /// Find all paths between two nodes within depth limit
    ///
    /// # Arguments
    /// * `from_id` - Starting node ID
    /// * `to_id` - Target node ID
    /// * `params` - Traversal parameters
    ///
    /// # Returns
    /// All paths found, sorted by length/weight
    async fn find_all_paths(
        &self,
        from_id: &str,
        to_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<GraphPath>>;

    /// Traverse from a node following specified patterns
    ///
    /// # Arguments
    /// * `start_id` - Starting node ID
    /// * `params` - Traversal parameters
    ///
    /// # Returns
    /// All reachable nodes within the specified constraints
    async fn traverse_from(
        &self,
        start_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<GraphNode>>;

    /// Find nodes that match specific criteria
    ///
    /// # Arguments
    /// * `labels` - Node labels to match (empty = any)
    /// * `properties` - Property filters
    ///
    /// # Returns
    /// Matching nodes
    async fn find_nodes(
        &self,
        labels: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<Vec<GraphNode>>;

    /// Find relationships that match specific criteria
    ///
    /// # Arguments
    /// * `relationship_types` - Relationship types to match (empty = any)
    /// * `properties` - Property filters
    ///
    /// # Returns
    /// Matching relationships
    async fn find_relationships(
        &self,
        relationship_types: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<Vec<GraphRelationship>>;
}

/// Advanced graph analytics operations
///
/// This trait provides methods for graph analysis, pattern recognition,
/// and intelligence extraction from graph structures.
#[async_trait]
pub trait GraphAnalytics: Send + Sync {
    /// Calculate centrality measures for nodes
    ///
    /// # Arguments
    /// * `node_ids` - Node IDs to analyze (empty = all nodes)
    /// * `centrality_type` - Type of centrality to calculate
    ///
    /// # Returns
    /// Map of node ID to centrality score
    async fn calculate_centrality(
        &self,
        node_ids: Vec<String>,
        centrality_type: CentralityType,
    ) -> TylResult<HashMap<String, f64>>;

    /// Detect communities/clusters in the graph
    ///
    /// # Arguments
    /// * `algorithm` - Clustering algorithm to use
    /// * `params` - Algorithm-specific parameters
    ///
    /// # Returns
    /// Map of node ID to community/cluster ID
    async fn detect_communities(
        &self,
        algorithm: ClusteringAlgorithm,
        params: HashMap<String, serde_json::Value>,
    ) -> TylResult<HashMap<String, String>>;

    /// Find frequently occurring patterns in the graph
    ///
    /// # Arguments
    /// * `pattern_size` - Size of patterns to find
    /// * `min_frequency` - Minimum frequency threshold
    ///
    /// # Returns
    /// Detected patterns with their frequencies
    async fn find_patterns(
        &self,
        pattern_size: usize,
        min_frequency: usize,
    ) -> TylResult<Vec<(GraphPath, usize)>>;

    /// Recommend new relationships based on graph structure
    ///
    /// # Arguments
    /// * `node_id` - Node to generate recommendations for
    /// * `recommendation_type` - Type of recommendations
    /// * `limit` - Maximum number of recommendations
    ///
    /// # Returns
    /// Recommended relationships with confidence scores
    /// (target_node_id, relationship_type, confidence)
    async fn recommend_relationships(
        &self,
        node_id: &str,
        recommendation_type: RecommendationType,
        limit: usize,
    ) -> TylResult<Vec<(String, String, f64)>>;
}

/// Query execution for custom graph operations
///
/// This trait allows execution of database-specific queries
/// for operations not covered by the standard interface.
#[async_trait]
pub trait GraphQueryExecutor: Send + Sync {
    /// Execute a custom query
    ///
    /// # Arguments
    /// * `query` - Graph query to execute
    ///
    /// # Returns
    /// Query results
    ///
    /// # Implementation Notes
    /// - Should validate query syntax
    /// - Should handle parameterized queries safely
    /// - Should respect transaction boundaries
    async fn execute_query(&self, query: GraphQuery) -> TylResult<QueryResult>;

    /// Execute a read-only query
    ///
    /// # Arguments
    /// * `query` - Read-only graph query
    ///
    /// # Returns
    /// Query results
    ///
    /// # Implementation Notes
    /// - Should enforce read-only constraint
    /// - Can be optimized for read operations
    async fn execute_read_query(&self, query: GraphQuery) -> TylResult<QueryResult>;

    /// Execute a write query within a transaction
    ///
    /// # Arguments
    /// * `query` - Write graph query
    ///
    /// # Returns
    /// Query results
    ///
    /// # Implementation Notes
    /// - Should ensure ACID properties
    /// - Should handle rollback on failure
    async fn execute_write_query(&self, query: GraphQuery) -> TylResult<QueryResult>;
}

/// Health checking for graph store
#[async_trait]
pub trait GraphHealth: Send + Sync {
    /// Check if the graph store is healthy and responsive
    ///
    /// # Returns
    /// * `Ok(true)` if healthy
    /// * `Ok(false)` if unhealthy but reachable
    /// * `Err(...)` if unreachable
    async fn is_healthy(&self) -> TylResult<bool>;

    /// Get detailed health information
    ///
    /// # Returns
    /// Health metrics and status information
    async fn health_check(&self) -> TylResult<HashMap<String, serde_json::Value>>;

    /// Get graph statistics
    ///
    /// # Returns
    /// Statistics like node count, relationship count, etc.
    async fn get_statistics(&self) -> TylResult<HashMap<String, serde_json::Value>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node_creation_should_work() {
        // TDD: Create a graph node with builder pattern
        let node = GraphNode::new()
            .with_label("Person")
            .with_property("name", serde_json::json!("Alice"))
            .with_property("age", serde_json::json!(30));

        assert_eq!(node.labels, vec!["Person"]);
        assert_eq!(node.properties.get("name"), Some(&serde_json::json!("Alice")));
        assert_eq!(node.properties.get("age"), Some(&serde_json::json!(30)));
    }

    #[test]
    fn test_graph_relationship_creation_should_work() {
        // TDD: Create a relationship with builder pattern
        let relationship = GraphRelationship::new("node1", "node2", "KNOWS")
            .with_property("since", serde_json::json!("2020-01-01"));

        assert_eq!(relationship.from_node_id, "node1");
        assert_eq!(relationship.to_node_id, "node2");
        assert_eq!(relationship.relationship_type, "KNOWS");
        assert_eq!(relationship.properties.get("since"), Some(&serde_json::json!("2020-01-01")));
    }

    #[test]
    fn test_traversal_params_default_should_have_sensible_values() {
        // TDD: Default traversal parameters should work for most cases
        let params = TraversalParams::default();

        assert_eq!(params.max_depth, Some(3));
        assert_eq!(params.limit, Some(100));
        assert!(matches!(params.direction, TraversalDirection::Both));
        assert!(params.relationship_types.is_empty());
        assert!(params.node_labels.is_empty());
    }

    #[test]
    fn test_graph_path_creation_should_calculate_length() {
        // TDD: Graph path should automatically calculate length
        let node1 = GraphNode::new().with_label("Person");
        let node2 = GraphNode::new().with_label("Person");
        let rel = GraphRelationship::new("1", "2", "KNOWS");

        let path = GraphPath::new()
            .add_node(node1)
            .add_relationship(rel)
            .add_node(node2);

        assert_eq!(path.length, 1);
        assert_eq!(path.nodes.len(), 2);
        assert_eq!(path.relationships.len(), 1);
    }

    #[test]
    fn test_serialization_should_preserve_data() {
        // TDD: All types should serialize/deserialize correctly
        let node = GraphNode::new()
            .with_label("Test")
            .with_property("key", serde_json::json!("value"));

        let serialized = serde_json::to_string(&node).unwrap();
        let deserialized: GraphNode = serde_json::from_str(&serialized).unwrap();

        assert_eq!(node.labels, deserialized.labels);
        assert_eq!(node.properties, deserialized.properties);
    }
}
