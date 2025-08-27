//! Graph types and data structures
//!
//! This module defines the core types used throughout the graph port,
//! including nodes, relationships, paths, and configuration structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a node in the graph with properties and labels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: String,
    /// Node labels/types (e.g., ["Person", "Employee"])
    pub labels: Vec<String>,
    /// Node properties as key-value pairs
    pub properties: HashMap<String, serde_json::Value>,
    /// When this node was created
    pub created_at: DateTime<Utc>,
    /// When this node was last updated
    pub updated_at: DateTime<Utc>,
}

impl GraphNode {
    /// Create a new graph node with a generated ID
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            labels: Vec::new(),
            properties: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new graph node with a specific ID
    pub fn with_id(id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            labels: Vec::new(),
            properties: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a label to this node
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Add multiple labels to this node
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels.extend(labels);
        self
    }

    /// Add a property to this node
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Add multiple properties to this node
    pub fn with_properties(mut self, properties: HashMap<String, serde_json::Value>) -> Self {
        self.properties.extend(properties);
        self
    }

    /// Update the node's timestamp
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

impl Default for GraphNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a relationship between two nodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphRelationship {
    /// Unique identifier for this relationship
    pub id: String,
    /// ID of the source node
    pub from_node_id: String,
    /// ID of the target node
    pub to_node_id: String,
    /// Relationship type (e.g., "KNOWS", "WORKS_FOR", "CONTAINS")
    pub relationship_type: String,
    /// Relationship properties
    pub properties: HashMap<String, serde_json::Value>,
    /// When this relationship was created
    pub created_at: DateTime<Utc>,
    /// When this relationship was last updated
    pub updated_at: DateTime<Utc>,
}

impl GraphRelationship {
    /// Create a new relationship
    pub fn new(
        from_node_id: impl Into<String>,
        to_node_id: impl Into<String>,
        relationship_type: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            from_node_id: from_node_id.into(),
            to_node_id: to_node_id.into(),
            relationship_type: relationship_type.into(),
            properties: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new relationship with a specific ID
    pub fn with_id(
        id: impl Into<String>,
        from_node_id: impl Into<String>,
        to_node_id: impl Into<String>,
        relationship_type: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            from_node_id: from_node_id.into(),
            to_node_id: to_node_id.into(),
            relationship_type: relationship_type.into(),
            properties: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a property to this relationship
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Add multiple properties to this relationship
    pub fn with_properties(mut self, properties: HashMap<String, serde_json::Value>) -> Self {
        self.properties.extend(properties);
        self
    }

    /// Update the relationship's timestamp
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

/// Result of a graph traversal or query
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphPath {
    /// Nodes in the path
    pub nodes: Vec<GraphNode>,
    /// Relationships in the path
    pub relationships: Vec<GraphRelationship>,
    /// Total path length (number of relationships)
    pub length: usize,
    /// Path weight/score if applicable
    pub weight: Option<f64>,
}

impl GraphPath {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            relationships: Vec::new(),
            length: 0,
            weight: None,
        }
    }

    /// Add a node to the path
    pub fn add_node(mut self, node: GraphNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add a relationship to the path and update length
    pub fn add_relationship(mut self, relationship: GraphRelationship) -> Self {
        self.relationships.push(relationship);
        self.length = self.relationships.len();
        self
    }

    /// Set the path weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Calculate path weight based on relationship count if not set
    pub fn calculate_weight(&mut self) {
        if self.weight.is_none() {
            self.weight = Some(self.length as f64);
        }
    }
}

impl Default for GraphPath {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameters for graph traversal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraversalParams {
    /// Maximum depth to traverse
    pub max_depth: Option<usize>,
    /// Relationship types to follow (empty = all)
    pub relationship_types: Vec<String>,
    /// Node labels to include (empty = all)
    pub node_labels: Vec<String>,
    /// Property filters for nodes
    pub node_filters: HashMap<String, serde_json::Value>,
    /// Property filters for relationships
    pub relationship_filters: HashMap<String, serde_json::Value>,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Direction of traversal
    pub direction: TraversalDirection,
}

impl TraversalParams {
    /// Create new traversal parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Add relationship type filter
    pub fn with_relationship_type(mut self, rel_type: impl Into<String>) -> Self {
        self.relationship_types.push(rel_type.into());
        self
    }

    /// Add node label filter
    pub fn with_node_label(mut self, label: impl Into<String>) -> Self {
        self.node_labels.push(label.into());
        self
    }

    /// Set traversal direction
    pub fn with_direction(mut self, direction: TraversalDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Add node property filter
    pub fn with_node_filter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.node_filters.insert(key.into(), value);
        self
    }

    /// Add relationship property filter
    pub fn with_relationship_filter(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.relationship_filters.insert(key.into(), value);
        self
    }
}

impl Default for TraversalParams {
    fn default() -> Self {
        Self {
            max_depth: Some(3),
            relationship_types: Vec::new(),
            node_labels: Vec::new(),
            node_filters: HashMap::new(),
            relationship_filters: HashMap::new(),
            limit: Some(100),
            direction: TraversalDirection::Both,
        }
    }
}

/// Direction for graph traversal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TraversalDirection {
    /// Follow outgoing relationships only
    Outgoing,
    /// Follow incoming relationships only
    Incoming,
    /// Follow both directions
    Both,
}

/// Query builder for complex graph queries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphQuery {
    /// Raw query string (database-specific)
    pub query: String,
    /// Query parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Whether this query modifies data
    pub is_write_query: bool,
}

impl GraphQuery {
    /// Create a new read query
    pub fn read(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            parameters: HashMap::new(),
            is_write_query: false,
        }
    }

    /// Create a new write query
    pub fn write(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            parameters: HashMap::new(),
            is_write_query: true,
        }
    }

    /// Add a parameter to the query
    pub fn with_parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Add multiple parameters to the query
    pub fn with_parameters(mut self, parameters: HashMap<String, serde_json::Value>) -> Self {
        self.parameters.extend(parameters);
        self
    }
}

/// Result of executing a graph query
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryResult {
    /// Result data as JSON
    pub data: Vec<HashMap<String, serde_json::Value>>,
    /// Query execution metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl QueryResult {
    /// Create a new query result
    pub fn new() -> Self {
        Self { data: Vec::new(), metadata: HashMap::new() }
    }

    /// Add a data row to the result
    pub fn add_row(mut self, row: HashMap<String, serde_json::Value>) -> Self {
        self.data.push(row);
        self
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl Default for QueryResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of centrality measures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CentralityType {
    /// Degree centrality (number of connections)
    Degree,
    /// Betweenness centrality (bridge importance)
    Betweenness,
    /// Closeness centrality (average distance to all nodes)
    Closeness,
    /// Eigenvector centrality (connection to important nodes)
    Eigenvector,
    /// PageRank centrality
    PageRank,
    /// Katz centrality
    Katz,
}

/// Clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusteringAlgorithm {
    /// Louvain community detection
    Louvain,
    /// Label propagation
    LabelPropagation,
    /// Leiden algorithm
    Leiden,
    /// Connected components
    ConnectedComponents,
}

/// Types of relationship recommendations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendationType {
    /// Similar nodes (collaborative filtering)
    SimilarNodes,
    /// Common neighbors
    CommonNeighbors,
    /// Structural equivalence
    StructuralEquivalence,
    /// Path-based similarity
    PathSimilarity,
}

// ===== NEW FEATURES - Multi-Graph and Advanced Capabilities =====

/// Graph information and metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GraphInfo {
    /// Graph identifier
    pub id: String,
    /// Graph name/description
    pub name: String,
    /// Graph metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// When this graph was created
    pub created_at: DateTime<Utc>,
    /// When this graph was last updated
    pub updated_at: DateTime<Utc>,
}

impl GraphInfo {
    /// Create a new graph info
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            name: name.into(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add metadata to the graph
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Transaction context for ACID operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransactionContext {
    /// Transaction ID
    pub id: String,
    /// Is read-only transaction
    pub read_only: bool,
    /// Transaction isolation level
    pub isolation_level: IsolationLevel,
    /// Transaction timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Transaction metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl TransactionContext {
    /// Create a new read-write transaction
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            read_only: false,
            isolation_level: IsolationLevel::ReadCommitted,
            timeout_seconds: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a read-only transaction
    pub fn read_only() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            read_only: true,
            isolation_level: IsolationLevel::ReadCommitted,
            timeout_seconds: None,
            metadata: HashMap::new(),
        }
    }

    /// Set transaction isolation level
    pub fn with_isolation(mut self, level: IsolationLevel) -> Self {
        self.isolation_level = level;
        self
    }

    /// Set transaction timeout
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }
}

impl Default for TransactionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction isolation levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IsolationLevel {
    /// Read uncommitted (lowest isolation)
    ReadUncommitted,
    /// Read committed (default for most databases)
    ReadCommitted,
    /// Repeatable read
    RepeatableRead,
    /// Serializable (highest isolation)
    Serializable,
}

/// Index configuration for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexConfig {
    /// Index name
    pub name: String,
    /// Index type
    pub index_type: IndexType,
    /// Properties to index
    pub properties: Vec<String>,
    /// Labels to index (for nodes) or relationship types (for relationships)
    pub labels_or_types: Vec<String>,
    /// Index options
    pub options: HashMap<String, serde_json::Value>,
}

impl IndexConfig {
    /// Create a new node property index
    pub fn node_property(
        name: impl Into<String>,
        labels: Vec<String>,
        properties: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            index_type: IndexType::NodeProperty,
            properties,
            labels_or_types: labels,
            options: HashMap::new(),
        }
    }

    /// Create a new relationship property index
    pub fn relationship_property(
        name: impl Into<String>,
        relationship_types: Vec<String>,
        properties: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            index_type: IndexType::RelationshipProperty,
            properties,
            labels_or_types: relationship_types,
            options: HashMap::new(),
        }
    }

    /// Create a full-text index
    pub fn fulltext(name: impl Into<String>, labels: Vec<String>, properties: Vec<String>) -> Self {
        Self {
            name: name.into(),
            index_type: IndexType::Fulltext,
            properties,
            labels_or_types: labels,
            options: HashMap::new(),
        }
    }

    /// Add option to the index
    pub fn with_option(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.options.insert(key.into(), value);
        self
    }
}

/// Types of indexes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    /// Standard property index for nodes
    NodeProperty,
    /// Standard property index for relationships
    RelationshipProperty,
    /// Full-text search index
    Fulltext,
    /// Vector similarity index
    Vector,
    /// Composite index (multiple properties)
    Composite,
}

/// Constraint configuration for data integrity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConstraintConfig {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Properties involved in constraint
    pub properties: Vec<String>,
    /// Labels (for nodes) or relationship types (for relationships)
    pub labels_or_types: Vec<String>,
    /// Constraint options
    pub options: HashMap<String, serde_json::Value>,
}

impl ConstraintConfig {
    /// Create a uniqueness constraint
    pub fn unique(name: impl Into<String>, labels: Vec<String>, properties: Vec<String>) -> Self {
        Self {
            name: name.into(),
            constraint_type: ConstraintType::Unique,
            properties,
            labels_or_types: labels,
            options: HashMap::new(),
        }
    }

    /// Create an existence constraint
    pub fn exists(name: impl Into<String>, labels: Vec<String>, properties: Vec<String>) -> Self {
        Self {
            name: name.into(),
            constraint_type: ConstraintType::Exists,
            properties,
            labels_or_types: labels,
            options: HashMap::new(),
        }
    }

    /// Add option to the constraint
    pub fn with_option(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.options.insert(key.into(), value);
        self
    }
}

/// Types of constraints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstraintType {
    /// Uniqueness constraint
    Unique,
    /// Property existence constraint
    Exists,
    /// Type constraint (property must be of specific type)
    Type,
    /// Range constraint (property must be within range)
    Range,
}

/// Aggregation function configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregationQuery {
    /// Function to apply
    pub function: AggregationFunction,
    /// Property to aggregate
    pub property: Option<String>,
    /// Group by properties
    pub group_by: Vec<String>,
    /// Filters to apply
    pub filters: HashMap<String, serde_json::Value>,
}

impl AggregationQuery {
    /// Create a COUNT aggregation
    pub fn count() -> Self {
        Self {
            function: AggregationFunction::Count,
            property: None,
            group_by: Vec::new(),
            filters: HashMap::new(),
        }
    }

    /// Create a SUM aggregation
    pub fn sum(property: impl Into<String>) -> Self {
        Self {
            function: AggregationFunction::Sum,
            property: Some(property.into()),
            group_by: Vec::new(),
            filters: HashMap::new(),
        }
    }

    /// Create an AVG aggregation
    pub fn avg(property: impl Into<String>) -> Self {
        Self {
            function: AggregationFunction::Avg,
            property: Some(property.into()),
            group_by: Vec::new(),
            filters: HashMap::new(),
        }
    }

    /// Add grouping
    pub fn group_by(mut self, property: impl Into<String>) -> Self {
        self.group_by.push(property.into());
        self
    }

    /// Add filter
    pub fn with_filter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.filters.insert(key.into(), value);
        self
    }
}

/// Types of aggregation functions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AggregationFunction {
    /// Count elements
    Count,
    /// Sum numeric values
    Sum,
    /// Average numeric values
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Collect values into array
    Collect,
    /// Standard deviation
    StdDev,
}

/// Result of aggregation query
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregationResult {
    /// Aggregated value
    pub value: serde_json::Value,
    /// Group by values (if any)
    pub groups: HashMap<String, serde_json::Value>,
    /// Result metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Weighted path for advanced pathfinding
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightedPath {
    /// Path nodes and relationships
    pub path: GraphPath,
    /// Total path weight
    pub total_weight: f64,
    /// Individual edge weights
    pub edge_weights: Vec<f64>,
    /// Weight calculation method used
    pub weight_method: WeightMethod,
}

impl WeightedPath {
    /// Create a weighted path
    pub fn new(path: GraphPath, edge_weights: Vec<f64>, method: WeightMethod) -> Self {
        let total_weight = match method {
            WeightMethod::Sum => edge_weights.iter().sum(),
            WeightMethod::Average => edge_weights.iter().sum::<f64>() / edge_weights.len() as f64,
            WeightMethod::Max => edge_weights.iter().cloned().fold(0.0, f64::max),
            WeightMethod::Min => edge_weights.iter().cloned().fold(f64::INFINITY, f64::min),
        };

        Self { path, total_weight, edge_weights, weight_method: method }
    }
}

/// Weight calculation methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WeightMethod {
    /// Sum all edge weights
    Sum,
    /// Average edge weights
    Average,
    /// Maximum edge weight
    Max,
    /// Minimum edge weight
    Min,
}

/// Temporal query parameters for time-based operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalQuery {
    /// Start time for query
    pub start_time: Option<DateTime<Utc>>,
    /// End time for query
    pub end_time: Option<DateTime<Utc>>,
    /// Temporal property to query on
    pub temporal_property: String,
    /// Temporal operation
    pub operation: TemporalOperation,
}

impl TemporalQuery {
    /// Create a query for data at specific time
    pub fn at(time: DateTime<Utc>, property: impl Into<String>) -> Self {
        Self {
            start_time: Some(time),
            end_time: Some(time),
            temporal_property: property.into(),
            operation: TemporalOperation::At,
        }
    }

    /// Create a query for data between times
    pub fn between(start: DateTime<Utc>, end: DateTime<Utc>, property: impl Into<String>) -> Self {
        Self {
            start_time: Some(start),
            end_time: Some(end),
            temporal_property: property.into(),
            operation: TemporalOperation::Between,
        }
    }

    /// Create a query for data before time
    pub fn before(time: DateTime<Utc>, property: impl Into<String>) -> Self {
        Self {
            start_time: None,
            end_time: Some(time),
            temporal_property: property.into(),
            operation: TemporalOperation::Before,
        }
    }

    /// Create a query for data after time
    pub fn after(time: DateTime<Utc>, property: impl Into<String>) -> Self {
        Self {
            start_time: Some(time),
            end_time: None,
            temporal_property: property.into(),
            operation: TemporalOperation::After,
        }
    }
}

/// Temporal query operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TemporalOperation {
    /// At specific time
    At,
    /// Between two times
    Between,
    /// Before specific time
    Before,
    /// After specific time
    After,
}

/// Bulk operation configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BulkOperation<T> {
    /// Items to process
    pub items: Vec<T>,
    /// Batch size for processing
    pub batch_size: usize,
    /// Continue on error
    pub continue_on_error: bool,
    /// Operation metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl<T> BulkOperation<T> {
    /// Create a new bulk operation
    pub fn new(items: Vec<T>) -> Self {
        Self {
            items,
            batch_size: 1000,
            continue_on_error: false,
            metadata: HashMap::new(),
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set error handling
    pub fn continue_on_error(mut self) -> Self {
        self.continue_on_error = true;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node_builder_pattern_should_work() {
        // TDD: Node builder pattern should create proper nodes
        let node = GraphNode::new()
            .with_label("Person")
            .with_label("Employee")
            .with_property("name", serde_json::json!("Alice"))
            .with_property("age", serde_json::json!(30));

        assert_eq!(node.labels, vec!["Person", "Employee"]);
        assert_eq!(node.properties.len(), 2);
        assert_eq!(node.properties.get("name"), Some(&serde_json::json!("Alice")));
        assert!(!node.id.is_empty());
    }

    #[test]
    fn test_graph_relationship_builder_pattern_should_work() {
        // TDD: Relationship builder pattern should create proper relationships
        let rel = GraphRelationship::new("node1", "node2", "KNOWS")
            .with_property("since", serde_json::json!("2020"))
            .with_property("strength", serde_json::json!(0.8));

        assert_eq!(rel.from_node_id, "node1");
        assert_eq!(rel.to_node_id, "node2");
        assert_eq!(rel.relationship_type, "KNOWS");
        assert_eq!(rel.properties.len(), 2);
        assert!(!rel.id.is_empty());
    }

    #[test]
    fn test_graph_path_builder_pattern_should_calculate_length() {
        // TDD: Path builder should calculate length correctly
        let node1 = GraphNode::new().with_label("A");
        let node2 = GraphNode::new().with_label("B");
        let node3 = GraphNode::new().with_label("C");
        let rel1 = GraphRelationship::new("1", "2", "TO");
        let rel2 = GraphRelationship::new("2", "3", "TO");

        let path = GraphPath::new()
            .add_node(node1)
            .add_relationship(rel1)
            .add_node(node2)
            .add_relationship(rel2)
            .add_node(node3)
            .with_weight(2.5);

        assert_eq!(path.length, 2);
        assert_eq!(path.nodes.len(), 3);
        assert_eq!(path.relationships.len(), 2);
        assert_eq!(path.weight, Some(2.5));
    }

    #[test]
    fn test_traversal_params_builder_pattern_should_work() {
        // TDD: TraversalParams builder should work correctly
        let params = TraversalParams::new()
            .with_max_depth(5)
            .with_relationship_type("KNOWS")
            .with_node_label("Person")
            .with_direction(TraversalDirection::Outgoing)
            .with_limit(50)
            .with_node_filter("active", serde_json::json!(true));

        assert_eq!(params.max_depth, Some(5));
        assert_eq!(params.relationship_types, vec!["KNOWS"]);
        assert_eq!(params.node_labels, vec!["Person"]);
        assert_eq!(params.direction, TraversalDirection::Outgoing);
        assert_eq!(params.limit, Some(50));
        assert_eq!(params.node_filters.get("active"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_graph_query_builder_pattern_should_work() {
        // TDD: GraphQuery builder should work correctly
        let query = GraphQuery::read("SELECT * FROM nodes")
            .with_parameter("limit", serde_json::json!(10))
            .with_parameter("offset", serde_json::json!(0));

        assert_eq!(query.query, "SELECT * FROM nodes");
        assert!(!query.is_write_query);
        assert_eq!(query.parameters.len(), 2);

        let write_query = GraphQuery::write("INSERT INTO persons (name) VALUES ($name)");
        assert!(write_query.is_write_query);
    }

    #[test]
    fn test_query_result_builder_pattern_should_work() {
        // TDD: QueryResult builder should work correctly
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), serde_json::json!("1"));
        row1.insert("name".to_string(), serde_json::json!("Alice"));

        let result = QueryResult::new()
            .add_row(row1)
            .with_metadata("total_count", serde_json::json!(1))
            .with_metadata("execution_time_ms", serde_json::json!(42));

        assert_eq!(result.data.len(), 1);
        assert_eq!(result.metadata.len(), 2);
    }

    #[test]
    fn test_enum_serialization_should_work() {
        // TDD: All enums should serialize/deserialize correctly
        let direction = TraversalDirection::Both;
        let serialized = serde_json::to_string(&direction).unwrap();
        let deserialized: TraversalDirection = serde_json::from_str(&serialized).unwrap();
        assert_eq!(direction, deserialized);

        let centrality = CentralityType::PageRank;
        let serialized = serde_json::to_string(&centrality).unwrap();
        let deserialized: CentralityType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(centrality, deserialized);
    }

    // ===== TESTS FOR NEW FEATURES =====

    #[test]
    fn test_graph_info_creation_should_work() {
        // TDD: GraphInfo should be creatable with metadata
        let graph_info = GraphInfo::new("social_graph", "Social Network Graph")
            .with_metadata("version", serde_json::json!("1.0"))
            .with_metadata("tenant", serde_json::json!("user_123"));

        assert_eq!(graph_info.id, "social_graph");
        assert_eq!(graph_info.name, "Social Network Graph");
        assert_eq!(graph_info.metadata.get("version"), Some(&serde_json::json!("1.0")));
        assert_eq!(graph_info.metadata.get("tenant"), Some(&serde_json::json!("user_123")));
    }

    #[test]
    fn test_transaction_context_creation_should_work() {
        // TDD: TransactionContext should support different configurations
        let tx_default = TransactionContext::new();
        assert!(!tx_default.read_only);
        assert_eq!(tx_default.isolation_level, IsolationLevel::ReadCommitted);

        let tx_readonly = TransactionContext::read_only()
            .with_isolation(IsolationLevel::Serializable)
            .with_timeout(30);

        assert!(tx_readonly.read_only);
        assert_eq!(tx_readonly.isolation_level, IsolationLevel::Serializable);
        assert_eq!(tx_readonly.timeout_seconds, Some(30));
    }

    #[test]
    fn test_index_config_creation_should_work() {
        // TDD: IndexConfig should support different index types
        let node_index = IndexConfig::node_property(
            "person_name_idx",
            vec!["Person".to_string()],
            vec!["name".to_string()],
        )
        .with_option("case_sensitive", serde_json::json!(false));

        assert_eq!(node_index.name, "person_name_idx");
        assert_eq!(node_index.index_type, IndexType::NodeProperty);
        assert_eq!(node_index.labels_or_types, vec!["Person"]);
        assert_eq!(node_index.properties, vec!["name"]);

        let fulltext_index = IndexConfig::fulltext(
            "content_search",
            vec!["Document".to_string()],
            vec!["title".to_string(), "content".to_string()],
        );

        assert_eq!(fulltext_index.index_type, IndexType::Fulltext);
        assert_eq!(fulltext_index.properties.len(), 2);
    }

    #[test]
    fn test_constraint_config_creation_should_work() {
        // TDD: ConstraintConfig should support different constraint types
        let unique_constraint = ConstraintConfig::unique(
            "person_email_unique",
            vec!["Person".to_string()],
            vec!["email".to_string()],
        );

        assert_eq!(unique_constraint.constraint_type, ConstraintType::Unique);
        assert_eq!(unique_constraint.properties, vec!["email"]);

        let exists_constraint = ConstraintConfig::exists(
            "person_name_required",
            vec!["Person".to_string()],
            vec!["name".to_string()],
        )
        .with_option("allow_empty", serde_json::json!(false));

        assert_eq!(exists_constraint.constraint_type, ConstraintType::Exists);
        assert!(exists_constraint.options.contains_key("allow_empty"));
    }

    #[test]
    fn test_aggregation_query_creation_should_work() {
        // TDD: AggregationQuery should support different functions
        let count_query = AggregationQuery::count()
            .group_by("label")
            .with_filter("active", serde_json::json!(true));

        assert_eq!(count_query.function, AggregationFunction::Count);
        assert_eq!(count_query.group_by, vec!["label"]);
        assert!(count_query.filters.contains_key("active"));

        let avg_query = AggregationQuery::avg("age").group_by("department");

        assert_eq!(avg_query.function, AggregationFunction::Avg);
        assert_eq!(avg_query.property, Some("age".to_string()));
        assert_eq!(avg_query.group_by, vec!["department"]);
    }

    #[test]
    fn test_weighted_path_creation_should_work() {
        // TDD: WeightedPath should calculate weights correctly
        let node1 = GraphNode::new().with_label("A");
        let node2 = GraphNode::new().with_label("B");
        let node3 = GraphNode::new().with_label("C");
        let rel1 = GraphRelationship::new("1", "2", "TO");
        let rel2 = GraphRelationship::new("2", "3", "TO");

        let path = GraphPath::new()
            .add_node(node1)
            .add_relationship(rel1)
            .add_node(node2)
            .add_relationship(rel2)
            .add_node(node3);

        let edge_weights = vec![2.5, 3.0];
        let weighted_path = WeightedPath::new(path, edge_weights.clone(), WeightMethod::Sum);

        assert_eq!(weighted_path.total_weight, 5.5);
        assert_eq!(weighted_path.edge_weights, edge_weights);
        assert_eq!(weighted_path.weight_method, WeightMethod::Sum);

        let avg_weighted_path = WeightedPath::new(
            weighted_path.path.clone(),
            edge_weights.clone(),
            WeightMethod::Average,
        );
        assert_eq!(avg_weighted_path.total_weight, 2.75);
    }

    #[test]
    fn test_temporal_query_creation_should_work() {
        // TDD: TemporalQuery should support different time operations
        use chrono::TimeZone;

        let time1 = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
        let time2 = Utc.with_ymd_and_hms(2024, 12, 31, 23, 59, 59).unwrap();

        let at_query = TemporalQuery::at(time1, "created_at");
        assert_eq!(at_query.operation, TemporalOperation::At);
        assert_eq!(at_query.start_time, Some(time1));
        assert_eq!(at_query.end_time, Some(time1));

        let between_query = TemporalQuery::between(time1, time2, "updated_at");
        assert_eq!(between_query.operation, TemporalOperation::Between);
        assert_eq!(between_query.start_time, Some(time1));
        assert_eq!(between_query.end_time, Some(time2));

        let before_query = TemporalQuery::before(time2, "expires_at");
        assert_eq!(before_query.operation, TemporalOperation::Before);
        assert_eq!(before_query.start_time, None);
        assert_eq!(before_query.end_time, Some(time2));

        let after_query = TemporalQuery::after(time1, "starts_at");
        assert_eq!(after_query.operation, TemporalOperation::After);
        assert_eq!(after_query.start_time, Some(time1));
        assert_eq!(after_query.end_time, None);
    }

    #[test]
    fn test_bulk_operation_configuration_should_work() {
        // TDD: BulkOperation should be configurable
        let nodes = vec![
            GraphNode::new().with_label("Person"),
            GraphNode::new().with_label("Person"),
            GraphNode::new().with_label("Person"),
        ];

        let bulk_op = BulkOperation::new(nodes.clone())
            .with_batch_size(100)
            .continue_on_error()
            .with_metadata("operation_id", serde_json::json!("bulk_001"));

        assert_eq!(bulk_op.items.len(), 3);
        assert_eq!(bulk_op.batch_size, 100);
        assert!(bulk_op.continue_on_error);
        assert_eq!(bulk_op.metadata.get("operation_id"), Some(&serde_json::json!("bulk_001")));
    }

    #[test]
    fn test_new_types_serialization_should_work() {
        // TDD: All new types should serialize/deserialize correctly
        let graph_info = GraphInfo::new("test_graph", "Test Graph");
        let serialized = serde_json::to_string(&graph_info).unwrap();
        let deserialized: GraphInfo = serde_json::from_str(&serialized).unwrap();
        assert_eq!(graph_info, deserialized);

        let tx_context = TransactionContext::new();
        let serialized = serde_json::to_string(&tx_context).unwrap();
        let deserialized: TransactionContext = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tx_context, deserialized);

        let isolation_level = IsolationLevel::Serializable;
        let serialized = serde_json::to_string(&isolation_level).unwrap();
        let deserialized: IsolationLevel = serde_json::from_str(&serialized).unwrap();
        assert_eq!(isolation_level, deserialized);
    }
}
