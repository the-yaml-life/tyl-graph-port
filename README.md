# TYL Graph Port

[![Rust](https://github.com/the-yaml-life/tyl-graph-port/workflows/CI/badge.svg)](https://github.com/the-yaml-life/tyl-graph-port/actions)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Crates.io](https://img.shields.io/crates/v/tyl-graph-port.svg)](https://crates.io/crates/tyl-graph-port)
[![Documentation](https://docs.rs/tyl-graph-port/badge.svg)](https://docs.rs/tyl-graph-port)

Graph database port for the TYL (The YAML Life) framework. This library defines the contract for graph database operations following hexagonal architecture principles with multi-graph support.

## Features

- **Multi-Graph Support**: Manage multiple isolated graphs within a single instance
- **Graph Storage**: CRUD operations for nodes and relationships
- **Graph Traversal**: Path finding, neighbor discovery, and graph exploration  
- **Graph Analytics**: Centrality calculations, community detection, pattern recognition
- **Transactions**: ACID transaction support with configurable isolation levels
- **Indexing**: Node property, fulltext, vector, and composite index management
- **Constraints**: Data integrity with unique, exists, type, and range constraints
- **Bulk Operations**: Batch processing and data import/export
- **Temporal Queries**: Time-based filtering for nodes and relationships
- **Query Execution**: Database-agnostic query interface
- **Health Monitoring**: Connection health and statistics
- **Error Handling**: Uses TYL error framework
- **Mock Implementation**: Complete in-memory implementation for testing
- **Async Support**: Full async/await support

## Architecture

This module follows hexagonal architecture:

- **Ports (Interfaces)**: Graph traits define the contract
- **Adapters**: Database-specific implementations
- **Domain Logic**: Graph operations independent of infrastructure

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
tyl-graph-port = "0.1"
```

### Basic Usage

```rust
use tyl_graph_port::{GraphStore, GraphNode, GraphRelationship, TraversalParams, GraphInfo, MultiGraphManager};
use tyl_errors::TylResult;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> TylResult<()> {
    // Use the mock implementation for testing
    #[cfg(feature = "mock")]
    {
        use tyl_graph_port::MockGraphStore;
        
        let store = MockGraphStore::new();
        let graph_id = "my-graph";
        
        // Create a graph
        let graph_info = GraphInfo {
            id: graph_id.to_string(),
            name: "My Graph".to_string(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.create_graph(graph_info).await?;
        
        // Create nodes
        let alice = GraphNode::new()
            .with_label("Person")
            .with_property("name", serde_json::json!("Alice"));
            
        let bob = GraphNode::new()
            .with_label("Person")
            .with_property("name", serde_json::json!("Bob"));
            
        let alice_id = store.create_node(graph_id, alice).await?;
        let bob_id = store.create_node(graph_id, bob).await?;
        
        // Create relationship
        let friendship = GraphRelationship::new(&alice_id, &bob_id, "KNOWS")
            .with_property("since", serde_json::json!("2020-01-01"));
            
        store.create_relationship(graph_id, friendship).await?;
        
        // Traverse the graph
        let neighbors = store.get_neighbors(graph_id, &alice_id, TraversalParams::default()).await?;
        println!("Alice has {} neighbors", neighbors.len());
    }
    
    Ok(())
}
```

## Core Traits

### MultiGraphManager

Interface for managing multiple graphs:

```rust
#[async_trait]
pub trait MultiGraphManager: Send + Sync {
    async fn create_graph(&self, graph_info: GraphInfo) -> TylResult<GraphInfo>;
    async fn list_graphs(&self) -> TylResult<Vec<GraphInfo>>;
    async fn delete_graph(&self, graph_id: &str) -> TylResult<()>;
    async fn graph_exists(&self, graph_id: &str) -> TylResult<bool>;
}
```

### GraphStore

The main storage interface for graph operations:

```rust
#[async_trait]
pub trait GraphStore: DatabaseLifecycle + MultiGraphManager + Send + Sync {
    async fn create_node(&self, graph_id: &str, node: GraphNode) -> TylResult<String>;
    async fn get_node(&self, graph_id: &str, id: &str) -> TylResult<Option<GraphNode>>;
    async fn update_node(&self, graph_id: &str, id: &str, properties: HashMap<String, serde_json::Value>) -> TylResult<()>;
    async fn delete_node(&self, graph_id: &str, id: &str) -> TylResult<()>;
    
    async fn create_relationship(&self, graph_id: &str, relationship: GraphRelationship) -> TylResult<String>;
    async fn get_relationship(&self, graph_id: &str, id: &str) -> TylResult<Option<GraphRelationship>>;
    async fn update_relationship(&self, graph_id: &str, id: &str, properties: HashMap<String, serde_json::Value>) -> TylResult<()>;
    async fn delete_relationship(&self, graph_id: &str, id: &str) -> TylResult<()>;
}
```

### GraphTraversal

Interface for exploring graph structure:

```rust
#[async_trait]
pub trait GraphTraversal: Send + Sync {
    async fn get_neighbors(&self, graph_id: &str, node_id: &str, params: TraversalParams) -> TylResult<Vec<(GraphNode, GraphRelationship)>>;
    async fn find_shortest_path(&self, graph_id: &str, from_id: &str, to_id: &str, params: TraversalParams) -> TylResult<Option<GraphPath>>;
    async fn find_all_paths(&self, graph_id: &str, from_id: &str, to_id: &str, params: TraversalParams) -> TylResult<Vec<GraphPath>>;
    async fn traverse_from(&self, graph_id: &str, start_id: &str, params: TraversalParams) -> TylResult<Vec<GraphNode>>;
}
```

### GraphAnalytics

Interface for graph analysis:

```rust
#[async_trait]
pub trait GraphAnalytics: Send + Sync {
    async fn calculate_centrality(&self, graph_id: &str, node_ids: Vec<String>, centrality_type: CentralityType) -> TylResult<HashMap<String, f64>>;
    async fn detect_communities(&self, graph_id: &str, algorithm: ClusteringAlgorithm, params: HashMap<String, serde_json::Value>) -> TylResult<HashMap<String, String>>;
    async fn find_patterns(&self, graph_id: &str, pattern_size: usize, min_frequency: usize) -> TylResult<Vec<(GraphPath, usize)>>;
    async fn recommend_relationships(&self, graph_id: &str, node_id: &str, recommendation_type: RecommendationType, limit: usize) -> TylResult<Vec<(String, String, f64)>>;
}
```

### Other Traits

Additional traits for specialized functionality:

- **GraphTransaction**: ACID transaction support
- **GraphIndexManager**: Index creation and management  
- **GraphConstraintManager**: Data integrity constraints
- **GraphBulkOperations**: Batch processing operations
- **GraphQueryExecutor**: Database-agnostic query execution
- **GraphHealth**: Health monitoring and statistics

## Core Types

### GraphNode

Represents a node in the graph:

```rust
let node = GraphNode::new()
    .with_label("Person")
    .with_label("Employee")
    .with_property("name", serde_json::json!("Alice"))
    .with_property("age", serde_json::json!(30));
```

### GraphRelationship

Represents a relationship between two nodes:

```rust
let relationship = GraphRelationship::new("alice_id", "bob_id", "KNOWS")
    .with_property("since", serde_json::json!("2020-01-01"))
    .with_property("strength", serde_json::json!(0.8));
```

### TraversalParams

Configure graph traversal behavior:

```rust
let params = TraversalParams::new()
    .with_max_depth(3)
    .with_relationship_type("KNOWS")
    .with_node_label("Person")
    .with_direction(TraversalDirection::Both)
    .with_limit(100);
```

## Mock Implementation

The library includes a full in-memory mock implementation for testing:

```rust
use tyl_graph_port::MockGraphStore;

let store = MockGraphStore::new();
// All traits are implemented and work with real data
```

## Error Handling

All operations return `TylResult<T>` which integrates with the TYL error framework:

```rust
use tyl_graph_port::graph_errors;

// Create specific graph errors
let error = graph_errors::node_not_found("node_123");
let error = graph_errors::invalid_query("Invalid query syntax");
```

## Features

- `default` - Basic functionality
- `mock` - Include the mock implementation

## Integration with TYL Framework

This port integrates seamlessly with other TYL framework modules:

- **tyl-errors**: Comprehensive error handling
- **tyl-db-core**: Database lifecycle management
- **tyl-config**: Configuration management
- **tyl-logging**: Structured logging
- **tyl-tracing**: Distributed tracing

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.rs` - Comprehensive example showing all features

Run an example:

```bash
cargo run --example basic_usage --features mock
```

## Testing

Run tests:

```bash
# Run all tests
cargo test

# Run only unit tests
cargo test --lib

# Run integration tests
cargo test --test integration_tests

# Run with all features
cargo test --all-features

# Run doc tests
cargo test --doc
```

## Contributing

Please see the main TYL framework repository for contribution guidelines.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.