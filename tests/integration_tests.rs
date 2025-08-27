//! Integration tests for tyl-graph-port
//!
//! These tests verify that all graph traits work correctly together
//! and integrate properly with the TYL error framework.

use tyl_errors::TylResult;

#[cfg(feature = "mock")]
use std::collections::HashMap;
#[cfg(feature = "mock")]
use std::sync::Arc;
#[cfg(feature = "mock")]
use tyl_graph_port::*;

#[cfg(feature = "mock")]
use tyl_graph_port::MockGraphStore;

const TEST_GRAPH_ID: &str = "test_graph";

/// Setup a test graph store with a default graph
#[cfg(feature = "mock")]
async fn setup_test_store() -> TylResult<MockGraphStore> {
    let store = MockGraphStore::new();
    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), serde_json::json!("Test Graph"));
    metadata.insert(
        "description".to_string(),
        serde_json::json!("A test graph for integration tests"),
    );

    let graph_info = GraphInfo {
        id: TEST_GRAPH_ID.to_string(),
        metadata,
        ..Default::default()
    };
    store.create_graph(graph_info).await?;
    Ok(store)
}

/// Test the complete graph storage workflow
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_complete_graph_storage_workflow() -> TylResult<()> {
    // TDD: Complete workflow from creation to deletion with multi-graph support
    let store = setup_test_store().await?;
    let graph_id = "test-graph";

    // First create the graph
    let graph_info = GraphInfo {
        id: graph_id.to_string(),
        name: "Test Graph".to_string(),
        metadata: HashMap::new(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.create_graph(graph_info).await.unwrap();

    // Test batch node creation
    let nodes = vec![
        GraphNode::new()
            .with_label("Person")
            .with_property("name", serde_json::json!("Alice"))
            .with_property("age", serde_json::json!(30)),
        GraphNode::new()
            .with_label("Person")
            .with_property("name", serde_json::json!("Bob"))
            .with_property("age", serde_json::json!(25)),
        GraphNode::new()
            .with_label("Company")
            .with_property("name", serde_json::json!("TechCorp"))
            .with_property("industry", serde_json::json!("Technology")),
    ];

    let results = store.create_nodes_batch(graph_id, nodes).await.unwrap();
    assert_eq!(results.len(), 3);

    let node_ids: Vec<String> = results.into_iter().map(|r| r.unwrap()).collect();

    // Create relationships
    let relationships = vec![
        GraphRelationship::new(&node_ids[0], &node_ids[1], "KNOWS")
            .with_property("since", serde_json::json!("2020-01-01")),
        GraphRelationship::new(&node_ids[0], &node_ids[2], "WORKS_FOR")
            .with_property("position", serde_json::json!("Developer")),
        GraphRelationship::new(&node_ids[1], &node_ids[2], "WORKS_FOR")
            .with_property("position", serde_json::json!("Designer")),
    ];

    for rel in relationships {
        let _rel_id = store.create_relationship(TEST_GRAPH_ID, rel).await.unwrap();
    }

    // Verify the graph structure
    assert_eq!(store.node_count(TEST_GRAPH_ID).await, 3);
    assert_eq!(store.relationship_count(TEST_GRAPH_ID).await, 3);

    // Test finding nodes by criteria
    let people = store
        .find_nodes(graph_id, vec!["Person".to_string()], HashMap::new())
        .await
        .unwrap();
    assert_eq!(people.len(), 2);

    let companies = store
        .find_nodes(graph_id, vec!["Company".to_string()], HashMap::new())
        .await
        .unwrap();
    assert_eq!(companies.len(), 1);

    // Test finding relationships by type
    let work_relationships = store
        .find_relationships(graph_id, vec!["WORKS_FOR".to_string()], HashMap::new())
        .await
        .unwrap();
    assert_eq!(work_relationships.len(), 2);

    // Cleanup
    for node_id in node_ids {
        store.delete_node(TEST_GRAPH_ID, &node_id).await.unwrap();
    }

    assert_eq!(store.node_count(TEST_GRAPH_ID).await, 0);
    assert_eq!(store.relationship_count(TEST_GRAPH_ID).await, 0);
    Ok(())
}

/// Test complex graph traversal scenarios
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_complex_graph_traversal() -> TylResult<()> {
    // TDD: Complex graph traversal with filters and constraints with multi-graph support
    let store = setup_test_store().await?;
    let graph_id = "traversal-graph";

    // First create the graph
    let graph_info = GraphInfo {
        id: graph_id.to_string(),
        name: "Traversal Test Graph".to_string(),
        metadata: HashMap::new(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.create_graph(graph_info).await.unwrap();

    // Create a more complex graph structure
    // A -> B -> C
    // |    |    |
    // v    v    v
    // D -> E -> F
    let nodes = vec![
        GraphNode::with_id("A")
            .with_label("Level1")
            .with_property("value", serde_json::json!(1)),
        GraphNode::with_id("B")
            .with_label("Level1")
            .with_property("value", serde_json::json!(2)),
        GraphNode::with_id("C")
            .with_label("Level1")
            .with_property("value", serde_json::json!(3)),
        GraphNode::with_id("D")
            .with_label("Level2")
            .with_property("value", serde_json::json!(4)),
        GraphNode::with_id("E")
            .with_label("Level2")
            .with_property("value", serde_json::json!(5)),
        GraphNode::with_id("F")
            .with_label("Level2")
            .with_property("value", serde_json::json!(6)),
    ];

    for node in nodes {
        store.create_node(TEST_GRAPH_ID, node).await.unwrap();
    }

    let relationships = vec![
        GraphRelationship::new("A", "B", "HORIZONTAL"),
        GraphRelationship::new("B", "C", "HORIZONTAL"),
        GraphRelationship::new("A", "D", "VERTICAL"),
        GraphRelationship::new("B", "E", "VERTICAL"),
        GraphRelationship::new("C", "F", "VERTICAL"),
        GraphRelationship::new("D", "E", "HORIZONTAL"),
        GraphRelationship::new("E", "F", "HORIZONTAL"),
    ];

    for rel in relationships {
        store.create_relationship(TEST_GRAPH_ID, rel).await.unwrap();
    }

    // Test filtered traversal - only horizontal relationships
    let horizontal_params = TraversalParams::new()
        .with_relationship_type("HORIZONTAL")
        .with_max_depth(3);

    let horizontal_neighbors = store
        .get_neighbors(graph_id, "A", horizontal_params.clone())
        .await
        .unwrap();
    assert_eq!(horizontal_neighbors.len(), 1);
    assert_eq!(horizontal_neighbors[0].0.id, "B");

    // Test path finding with filters
    let path_horizontal = store
        .find_shortest_path(graph_id, "A", "C", horizontal_params)
        .await
        .unwrap();
    assert!(path_horizontal.is_some());
    let path = path_horizontal.unwrap();
    assert_eq!(path.length, 2);
    assert_eq!(path.nodes[0].id, "A");
    assert_eq!(path.nodes[1].id, "B");
    assert_eq!(path.nodes[2].id, "C");

    // Test bidirectional traversal
    let bidirectional_params = TraversalParams::new()
        .with_direction(TraversalDirection::Both)
        .with_max_depth(2);

    let reachable_from_e = store
        .traverse_from(graph_id, "E", bidirectional_params)
        .await
        .unwrap();
    assert!(reachable_from_e.len() >= 4); // E can reach multiple nodes

    // Test label filtering
    let level1_params = TraversalParams::new()
        .with_node_label("Level1")
        .with_max_depth(3);

    let level1_traversal = store
        .traverse_from(graph_id, "A", level1_params)
        .await
        .unwrap();
    let level1_count = level1_traversal
        .iter()
        .filter(|node| node.labels.contains(&"Level1".to_string()))
        .count();
    assert!(level1_count > 0);
}

/// Test graph analytics functionality
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_graph_analytics_integration() -> TylResult<()> {
    // TDD: Graph analytics with real graph data with multi-graph support
    let store = setup_test_store().await?;
    let graph_id = "analytics-graph";

    // First create the graph
    let graph_info = GraphInfo {
        id: graph_id.to_string(),
        name: "Analytics Test Graph".to_string(),
        metadata: HashMap::new(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.create_graph(graph_info).await.unwrap();

    // Create a star graph: central node connected to multiple others
    let center_node = GraphNode::with_id("center")
        .with_label("Hub")
        .with_property("type", serde_json::json!("central"));

    store.create_node(TEST_GRAPH_ID, center_node).await.unwrap();

    let spoke_nodes = (1..=5)
        .map(|i| {
            GraphNode::with_id(format!("spoke_{i}"))
                .with_label("Spoke")
                .with_property("number", serde_json::json!(i))
        })
        .collect::<Vec<_>>();

    for node in spoke_nodes {
        store.create_node(TEST_GRAPH_ID, node).await.unwrap();
    }

    // Connect all spokes to center
    for i in 1..=5 {
        let rel = GraphRelationship::new("center", format!("spoke_{i}"), "CONNECTS_TO");
        store.create_relationship(TEST_GRAPH_ID, rel).await.unwrap();
    }

    // Test centrality calculation
    let centrality = store
        .calculate_centrality(graph_id, vec!["center".to_string()], CentralityType::Degree)
        .await
        .unwrap();

    assert!(centrality.contains_key("center"));
    assert_eq!(centrality.get("center"), Some(&5.0)); // Center has degree 5

    // Test community detection
    let communities = store
        .detect_communities(graph_id, ClusteringAlgorithm::Louvain, HashMap::new())
        .await
        .unwrap();

    assert!(communities.contains_key("center"));
    assert_eq!(communities.len(), 6); // All nodes should be assigned to communities

    // Test relationship recommendations
    let recommendations = store
        .recommend_relationships(graph_id, "spoke_1", RecommendationType::CommonNeighbors, 3)
        .await
        .unwrap();

    // Should recommend connections to other spokes (they share the center as common neighbor)
    assert!(!recommendations.is_empty());
    assert!(recommendations.len() <= 3);
}

/// Test query execution functionality
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_query_execution_integration() -> TylResult<()> {
    // TDD: Query execution with proper validation with multi-graph support
    let store = setup_test_store().await?;
    let graph_id = "query-graph";

    // First create the graph
    let graph_info = GraphInfo {
        id: graph_id.to_string(),
        name: "Query Test Graph".to_string(),
        metadata: HashMap::new(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.create_graph(graph_info).await.unwrap();

    // Test read query execution
    let read_query = GraphQuery::read("SELECT * FROM nodes LIMIT 10")
        .with_parameter("limit", serde_json::json!(10));

    let read_result = store
        .execute_read_query(graph_id, read_query)
        .await
        .unwrap();
    assert!(read_result.metadata.contains_key("executed_at"));

    // Test write query execution
    let write_query = GraphQuery::write("INSERT INTO test_nodes (name) VALUES ($name)")
        .with_parameter("name", serde_json::json!("test"));

    let write_result = store
        .execute_write_query(graph_id, write_query)
        .await
        .unwrap();
    assert!(write_result.metadata.contains_key("executed_at"));

    // Test query validation - should fail when using wrong method
    let write_query_wrong = GraphQuery::write("INSERT INTO test_nodes DEFAULT VALUES");
    let result = store.execute_read_query(graph_id, write_query_wrong).await;
    assert!(result.is_err());

    let read_query_wrong = GraphQuery::read("SELECT * FROM nodes");
    let result = store.execute_write_query(graph_id, read_query_wrong).await;
    assert!(result.is_err());
}

/// Test health monitoring functionality
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_health_monitoring_integration() -> TylResult<()> {
    // TDD: Health monitoring with comprehensive checks
    let store = setup_test_store().await?;

    // Add some test data
    let node = GraphNode::new()
        .with_label("TestNode")
        .with_property("test", serde_json::json!(true));
    store.create_node(TEST_GRAPH_ID, node).await.unwrap();

    // Test basic health check
    let is_healthy = store.is_healthy().await.unwrap();
    assert!(is_healthy);

    // Test detailed health information
    let health_info = GraphHealth::health_check(&store).await.unwrap();
    assert_eq!(health_info.get("status"), Some(&serde_json::json!("healthy")));
    assert_eq!(health_info.get("type"), Some(&serde_json::json!("mock")));
    assert!(health_info.contains_key("uptime_seconds"));
    assert!(health_info.contains_key("version"));

    // Test statistics with real data
    let stats = store.get_all_statistics(TEST_GRAPH_ID).await.unwrap();
    assert_eq!(stats.get("node_count"), Some(&serde_json::json!(1)));
    assert_eq!(stats.get("relationship_count"), Some(&serde_json::json!(0)));

    // Verify label statistics
    let labels = stats.get("labels").unwrap();
    let labels_map: HashMap<String, usize> = serde_json::from_value(labels.clone()).unwrap();
    assert_eq!(labels_map.get("TestNode"), Some(&1));
}

/// Test error handling and edge cases
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_error_handling_integration() -> TylResult<()> {
    // TDD: Proper error handling throughout the system
    let store = setup_test_store().await?;

    // Test getting non-existent node
    let result = store.get_node(TEST_GRAPH_ID, "non_existent").await.unwrap();
    assert!(result.is_none());

    // Test updating non-existent node
    let mut updates = HashMap::new();
    updates.insert("key".to_string(), serde_json::json!("value"));
    let result = store
        .update_node(TEST_GRAPH_ID, "non_existent", updates)
        .await;
    assert!(result.is_err());

    // Test creating relationship with non-existent nodes
    let rel = GraphRelationship::new("non_existent_1", "non_existent_2", "TEST");
    let result = store.create_relationship(TEST_GRAPH_ID, rel).await;
    assert!(result.is_err());

    // Test updating non-existent relationship
    let mut updates = HashMap::new();
    updates.insert("key".to_string(), serde_json::json!("value"));
    let result = store.update_relationship("non_existent", updates).await;
    assert!(result.is_err());

    // Test traversal from non-existent node
    let result = store
        .get_neighbors("non_existent", TraversalParams::default())
        .await;
    assert!(result.is_err());

    // Test path finding with non-existent nodes
    let result = store
        .find_shortest_path("non_existent_1", "non_existent_2", TraversalParams::default())
        .await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());

    // Test analytics on non-existent node
    let result = store
        .recommend_relationships("non_existent", RecommendationType::SimilarNodes, 5)
        .await;
    assert!(result.is_err());
}

/// Test performance and scalability aspects
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_performance_and_scalability() -> TylResult<()> {
    // TDD: Basic performance characteristics
    let store = setup_test_store().await?;

    // Create a moderate number of nodes and relationships
    let node_count = 100;
    let mut node_ids = Vec::new();

    // Batch create nodes
    let nodes: Vec<GraphNode> = (0..node_count)
        .map(|i| {
            GraphNode::new()
                .with_label("TestNode")
                .with_property("index", serde_json::json!(i))
        })
        .collect();

    let start = std::time::Instant::now();
    let results = store.create_nodes_batch(nodes).await.unwrap();
    let creation_time = start.elapsed();

    // Should complete reasonably quickly
    assert!(creation_time.as_millis() < 1000);

    for result in results {
        node_ids.push(result.unwrap());
    }

    // Create relationships
    let relationship_count = 200;
    for i in 0..relationship_count {
        let from_idx = i % node_count;
        let to_idx = (i + 1) % node_count;
        let rel = GraphRelationship::new(&node_ids[from_idx], &node_ids[to_idx], "CONNECTS");
        store.create_relationship(TEST_GRAPH_ID, rel).await.unwrap();
    }

    // Test traversal performance
    let start = std::time::Instant::now();
    let _neighbors = store
        .get_neighbors(&node_ids[0], TraversalParams::default())
        .await
        .unwrap();
    let traversal_time = start.elapsed();

    // Should complete reasonably quickly
    assert!(traversal_time.as_millis() < 100);

    // Test path finding performance
    let start = std::time::Instant::now();
    let _path = store
        .find_shortest_path(&node_ids[0], &node_ids[10], TraversalParams::default())
        .await
        .unwrap();
    let pathfinding_time = start.elapsed();

    // Should complete reasonably quickly
    assert!(pathfinding_time.as_millis() < 500);

    // Verify final counts
    assert_eq!(store.node_count().await, node_count);
    assert_eq!(store.relationship_count().await, relationship_count);
}

/// Test serialization and data consistency
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_serialization_consistency() -> TylResult<()> {
    // TDD: Data consistency through serialization
    let store = setup_test_store().await?;

    // Create complex node with various data types
    let node = GraphNode::new()
        .with_label("ComplexNode")
        .with_property("string_value", serde_json::json!("test string"))
        .with_property("number_value", serde_json::json!(42))
        .with_property("float_value", serde_json::json!(std::f64::consts::PI))
        .with_property("boolean_value", serde_json::json!(true))
        .with_property("array_value", serde_json::json!(["a", "b", "c"]))
        .with_property("object_value", serde_json::json!({"nested": "value"}));

    let node_id = store
        .create_node(TEST_GRAPH_ID, node.clone())
        .await
        .unwrap();

    // Retrieve and verify data integrity
    let retrieved = store
        .get_node(TEST_GRAPH_ID, &node_id)
        .await
        .unwrap()
        .unwrap();

    // Serialize and deserialize to test consistency
    let serialized = serde_json::to_string(&retrieved).unwrap();
    let deserialized: GraphNode = serde_json::from_str(&serialized).unwrap();

    assert_eq!(retrieved.labels, deserialized.labels);
    assert_eq!(retrieved.properties, deserialized.properties);
    assert_eq!(retrieved.id, deserialized.id);

    // Test relationship serialization
    let node2_id = store
        .create_node(GraphNode::new().with_label("SimpleNode"))
        .await
        .unwrap();
    let rel = GraphRelationship::new(&node_id, &node2_id, "COMPLEX_RELATIONSHIP").with_property(
        "metadata",
        serde_json::json!({"version": 1, "tags": ["test", "integration"]}),
    );

    let rel_id = store.create_relationship(TEST_GRAPH_ID, rel).await.unwrap();
    let retrieved_rel = store
        .get_relationship(TEST_GRAPH_ID, &rel_id)
        .await
        .unwrap()
        .unwrap();

    let rel_serialized = serde_json::to_string(&retrieved_rel).unwrap();
    let rel_deserialized: GraphRelationship = serde_json::from_str(&rel_serialized).unwrap();

    assert_eq!(retrieved_rel.relationship_type, rel_deserialized.relationship_type);
    assert_eq!(retrieved_rel.properties, rel_deserialized.properties);
    assert_eq!(retrieved_rel.from_node_id, rel_deserialized.from_node_id);
    assert_eq!(retrieved_rel.to_node_id, rel_deserialized.to_node_id);
}

/// Test concurrent access patterns
#[cfg(feature = "mock")]
#[tokio::test]
async fn test_concurrent_access() -> TylResult<()> {
    // TDD: Basic concurrent access safety
    let store = std::sync::Arc::new(MockGraphStore::new());

    // Create initial node
    let initial_node = GraphNode::with_id("shared_node")
        .with_label("SharedNode")
        .with_property("counter", serde_json::json!(0));
    store
        .create_node(TEST_GRAPH_ID, initial_node)
        .await
        .unwrap();

    // Spawn multiple tasks that update the same node
    let mut handles = Vec::new();

    for i in 0..10 {
        let store_clone = Arc::clone(&store);
        let handle = tokio::spawn(async move {
            let mut updates = HashMap::new();
            updates.insert("counter".to_string(), serde_json::json!(i));
            store_clone.update_node("shared_node", updates).await
        });
        handles.push(handle);
    }

    // Wait for all updates to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // Verify node still exists and is accessible
    let final_node = store.get_node(TEST_GRAPH_ID, "shared_node").await.unwrap();
    assert!(final_node.is_some());
}

/// Test integration with TYL error framework
#[tokio::test]
async fn test_tyl_error_integration() -> TylResult<()> {
    // TDD: Proper integration with TYL error types

    // Test error creation and categorization
    let not_found_error = tyl_errors::TylError::not_found("graph_node", "test_id");
    let error_message = format!("{not_found_error}");
    println!("Error message: {error_message}");
    assert!(error_message.contains("not found") || error_message.contains("Not found"));

    let validation_error =
        tyl_errors::TylError::validation("graph_property", "Invalid property value");
    assert!(format!("{validation_error}").contains("Validation error"));

    // Test TylResult usage
    fn mock_operation_that_fails() -> TylResult<String> {
        Err(tyl_errors::TylError::database("Connection failed"))
    }

    let result = mock_operation_that_fails();
    assert!(result.is_err());

    match result {
        Err(e) => assert!(format!("{e}").contains("Connection failed")),
        Ok(_) => panic!("Expected error"),
    }
}
