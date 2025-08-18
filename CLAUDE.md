# CLAUDE.md - TYL Graph Port

## 📋 **Module Context**

**tyl-graph-port** is a core TYL framework module that defines the interface contracts for graph database operations. This module follows hexagonal architecture patterns to provide database-agnostic graph functionality.

## 🎯 **Module Purpose**

This module provides:

- **Graph Storage Interfaces**: CRUD operations for nodes and relationships
- **Graph Traversal Interfaces**: Path finding, neighbor discovery, graph exploration
- **Graph Analytics Interfaces**: Centrality calculations, community detection, pattern recognition
- **Query Execution Interfaces**: Custom query support for database-specific operations
- **Health Monitoring Interfaces**: Connection health and statistics
- **Mock Implementation**: Complete in-memory implementation for testing

## 🏗️ **Architecture Overview**

### **Hexagonal Architecture Implementation**

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Core                        │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ GraphStore    │  │ GraphTraversal│  │ GraphAnalytics  │  │
│  │ (Ports)       │  │ (Ports)       │  │ (Ports)         │  │
│  └───────────────┘  └──────────────┘  └─────────────────┘  │
│            │                 │                 │            │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ MockGraphStore│  │ Neo4jAdapter │  │ FalkorDBAdapter │  │
│  │ (Adapter)     │  │ (Adapter)    │  │ (Adapter)       │  │
│  └───────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### **Core Traits**

- **GraphStore**: Primary storage operations (CRUD for nodes/relationships)
- **GraphTraversal**: Graph exploration and path finding
- **GraphAnalytics**: Advanced graph analysis and intelligence
- **GraphQueryExecutor**: Custom query execution
- **GraphHealth**: Health monitoring and statistics

## 📊 **Data Model**

### **Core Types**

```rust
pub struct GraphNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub struct GraphRelationship {
    pub id: String,
    pub from_node_id: String,
    pub to_node_id: String,
    pub relationship_type: String,
    pub properties: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub struct GraphPath {
    pub nodes: Vec<GraphNode>,
    pub relationships: Vec<GraphRelationship>,
    pub length: usize,
    pub weight: Option<f64>,
}
```

### **Configuration Types**

```rust
pub struct TraversalParams {
    pub max_depth: Option<usize>,
    pub relationship_types: Vec<String>,
    pub node_labels: Vec<String>,
    pub node_filters: HashMap<String, serde_json::Value>,
    pub relationship_filters: HashMap<String, serde_json::Value>,
    pub limit: Option<usize>,
    pub direction: TraversalDirection,
}

pub enum TraversalDirection {
    Outgoing,
    Incoming,
    Both,
}
```

## 🔧 **Key Implementation Details**

### **Builder Patterns**

All core types use builder patterns for easy construction:

```rust
let node = GraphNode::new()
    .with_label("Person")
    .with_property("name", serde_json::json!("Alice"))
    .with_property("age", serde_json::json!(30));

let relationship = GraphRelationship::new("node1", "node2", "KNOWS")
    .with_property("since", serde_json::json!("2020-01-01"));

let params = TraversalParams::new()
    .with_max_depth(3)
    .with_relationship_type("KNOWS")
    .with_direction(TraversalDirection::Both);
```

### **Error Handling Integration**

Uses TYL error framework with graph-specific error helpers:

```rust
pub mod graph_errors {
    pub fn node_not_found(id: impl Into<String>) -> TylError;
    pub fn relationship_not_found(id: impl Into<String>) -> TylError;
    pub fn invalid_query(msg: impl Into<String>) -> TylError;
    pub fn traversal_error(msg: impl Into<String>) -> TylError;
    pub fn analytics_error(msg: impl Into<String>) -> TylError;
}
```

### **Mock Implementation**

Complete in-memory implementation suitable for:
- Unit testing
- Integration testing
- Development and prototyping
- Lightweight applications

The mock provides:
- Thread-safe concurrent access with `RwLock`
- Full implementation of all traits
- Realistic graph algorithms (BFS for shortest path)
- Performance characteristics suitable for testing

## 🧪 **Testing Strategy**

### **Test Coverage**

- **Unit Tests**: All core types and their builders
- **Integration Tests**: Complete workflow testing
- **Performance Tests**: Scalability characteristics
- **Concurrency Tests**: Thread-safety validation
- **Error Tests**: Comprehensive error scenarios
- **Serialization Tests**: Data consistency validation

### **Mock Usage in Tests**

```rust
#[tokio::test]
async fn test_graph_workflow() {
    let store = MockGraphStore::new();
    
    // Test complete workflow
    let node_id = store.create_node(node).await?;
    let neighbors = store.get_neighbors(&node_id, params).await?;
    // ... rest of test
}
```

## 🔗 **TYL Framework Integration**

### **Dependencies**

- **tyl-errors**: Error handling and result types
- **tyl-db-core**: Database lifecycle management
- **async-trait**: Async trait support
- **serde**: Serialization support
- **chrono**: Date/time handling
- **uuid**: Unique identifier generation

### **Re-exports**

The module re-exports TYL framework types for convenience:

```rust
pub use tyl_errors::{TylError, TylResult};
pub use tyl_db_core::{
    DatabaseLifecycle, DatabaseResult, HealthStatus, HealthCheckResult,
    DatabaseLogger, DatabaseTracer, // ... etc
};
```

## 🚀 **Usage Patterns**

### **Adapter Implementation**

When implementing a real graph database adapter:

```rust
pub struct Neo4jGraphStore {
    driver: Driver,
}

#[async_trait]
impl GraphStore for Neo4jGraphStore {
    async fn create_node(&self, node: GraphNode) -> TylResult<String> {
        // Convert to Cypher query
        // Execute via Neo4j driver
        // Return node ID
    }
    
    // ... implement other methods
}

// Also implement GraphTraversal, GraphAnalytics, etc.
```

### **Application Usage**

```rust
// Use any implementation through the trait
async fn process_graph<T>(store: &T) -> TylResult<()>
where
    T: GraphStore + GraphTraversal + GraphAnalytics,
{
    let nodes = store.find_nodes(labels, filters).await?;
    let centrality = store.calculate_centrality(node_ids, CentralityType::Degree).await?;
    // ... work with results
}
```

## 📝 **Development Guidelines**

### **When Extending This Module**

1. **Add New Traits**: For new categories of graph operations
2. **Extend Existing Traits**: For additional operations in existing categories
3. **Add New Types**: For new data structures or configuration
4. **Update Mock**: Always update mock implementation for new functionality

### **What NOT to Add**

- **Database-specific implementations** (these go in separate adapter crates)
- **Business logic** (this is infrastructure, not domain logic)
- **Configuration management** (use tyl-config)
- **Logging** (use tyl-logging)

### **Error Handling**

Always use `TylResult<T>` and the provided error helpers:

```rust
// ✅ Correct
async fn my_operation(&self) -> TylResult<String> {
    match some_condition {
        true => Ok("success".to_string()),
        false => Err(graph_errors::node_not_found("node_123")),
    }
}

// ❌ Wrong - don't create custom error types
async fn bad_operation(&self) -> Result<String, MyCustomError> {
    // ...
}
```

## 🔍 **Common Patterns**

### **Batch Operations**

```rust
// Use batch methods when available
let results = store.create_nodes_batch(nodes).await?;

// Handle mixed success/failure results
for result in results {
    match result {
        Ok(node_id) => println!("Created: {}", node_id),
        Err(e) => eprintln!("Failed: {}", e),
    }
}
```

### **Traversal with Filters**

```rust
let params = TraversalParams::new()
    .with_relationship_type("WORKS_FOR")
    .with_node_label("Person")
    .with_node_filter("active", serde_json::json!(true))
    .with_max_depth(2);

let reachable = store.traverse_from(&start_id, params).await?;
```

### **Analytics Workflows**

```rust
// Calculate centrality for important nodes
let centrality = store.calculate_centrality(
    key_node_ids,
    CentralityType::Betweenness
).await?;

// Find communities
let communities = store.detect_communities(
    ClusteringAlgorithm::Louvain,
    HashMap::new()
).await?;

// Get recommendations
let recommendations = store.recommend_relationships(
    &user_id,
    RecommendationType::CommonNeighbors,
    10
).await?;
```

## 🎯 **Quality Standards**

This module maintains high standards:

- **100% test coverage** for all public APIs
- **Comprehensive documentation** with examples
- **Performance benchmarks** for all algorithms
- **Thread safety** where applicable
- **Async-first design** throughout
- **Integration with TYL ecosystem**

The mock implementation serves as both a reference implementation and a complete testing solution, ensuring that all adapters follow the same behavioral patterns.