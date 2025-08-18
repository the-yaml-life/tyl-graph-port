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
        let query = GraphQuery::read("MATCH (n) RETURN n")
            .with_parameter("limit", serde_json::json!(10))
            .with_parameter("offset", serde_json::json!(0));

        assert_eq!(query.query, "MATCH (n) RETURN n");
        assert!(!query.is_write_query);
        assert_eq!(query.parameters.len(), 2);

        let write_query = GraphQuery::write("CREATE (n:Person {name: $name})");
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
}
