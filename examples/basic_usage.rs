//! Basic usage example for tyl-graph-port
//!
//! This example demonstrates the core functionality of the graph port,
//! including creating nodes and relationships, traversing the graph,
//! and performing basic analytics.

use tyl_errors::TylResult;

#[cfg(feature = "mock")]
use std::collections::HashMap;
#[cfg(feature = "mock")]
use tyl_graph_port::*;

#[cfg(feature = "mock")]
use tyl_graph_port::MockGraphStore;

const TEST_GRAPH_ID: &str = "test_graph";

#[tokio::main]
async fn main() -> TylResult<()> {
    println!("🚀 TYL Graph Port - Basic Usage Example");
    println!("=======================================");

    #[cfg(feature = "mock")]
    {
        // Create a mock graph store for demonstration
        let store = MockGraphStore::new();

        // Create a test graph
        let graph_info = GraphInfo {
            name: "Test Graph".to_string(),
            description: Some("A test graph for demonstrations".to_string()),
            ..Default::default()
        };
        store.create_graph(TEST_GRAPH_ID, graph_info).await?;

        // Demonstrate the complete workflow
        basic_node_operations(&store).await?;
        basic_relationship_operations(&store).await?;
        graph_traversal_examples(&store).await?;
        graph_analytics_examples(&store).await?;
        query_execution_examples(&store).await?;
        health_monitoring_examples(&store).await?;

        println!("\n✅ All examples completed successfully!");
    }

    #[cfg(not(feature = "mock"))]
    {
        println!("❌ Mock feature not enabled. Run with: cargo run --example basic_usage --features mock");
    }

    Ok(())
}

#[cfg(feature = "mock")]
async fn basic_node_operations(store: &MockGraphStore) -> TylResult<()> {
    println!("\n📍 Basic Node Operations");
    println!("-----------------------");

    // Create nodes with builder pattern
    let person1 = GraphNode::new()
        .with_label("Person")
        .with_label("Employee")
        .with_property("name", serde_json::json!("Alice Smith"))
        .with_property("age", serde_json::json!(30))
        .with_property("department", serde_json::json!("Engineering"))
        .with_property("active", serde_json::json!(true));

    let person2 = GraphNode::new()
        .with_label("Person")
        .with_label("Employee")
        .with_property("name", serde_json::json!("Bob Johnson"))
        .with_property("age", serde_json::json!(28))
        .with_property("department", serde_json::json!("Design"))
        .with_property("active", serde_json::json!(true));

    let company = GraphNode::new()
        .with_label("Company")
        .with_property("name", serde_json::json!("TechCorp Inc."))
        .with_property("industry", serde_json::json!("Technology"))
        .with_property("founded", serde_json::json!(2010))
        .with_property("employees", serde_json::json!(150));

    // Create nodes individually
    let alice_id = store.create_node(TEST_GRAPH_ID, person1).await?;
    let bob_id = store.create_node(TEST_GRAPH_ID, person2).await?;
    let company_id = store.create_node(TEST_GRAPH_ID, company).await?;

    println!("✅ Created nodes:");
    println!("   - Alice (Person/Employee): {alice_id}");
    println!("   - Bob (Person/Employee): {bob_id}");
    println!("   - TechCorp (Company): {company_id}");

    // Retrieve and display node information
    if let Some(alice) = store.get_node(TEST_GRAPH_ID, &alice_id).await? {
        println!("📄 Alice's details:");
        println!("   - Labels: {:?}", alice.labels);
        let name = alice.properties.get("name").unwrap();
        let department = alice.properties.get("department").unwrap();
        println!("   - Name: {name}");
        println!("   - Department: {department}");
    }

    // Update a node
    let mut updates = HashMap::new();
    updates.insert("age".to_string(), serde_json::json!(31));
    updates.insert("promotion_date".to_string(), serde_json::json!("2024-01-01"));
    store.update_node(TEST_GRAPH_ID, &alice_id, updates).await?;

    println!("✅ Updated Alice's age and added promotion date");

    // Batch create additional nodes
    let additional_employees = vec![
        GraphNode::new()
            .with_label("Person")
            .with_label("Employee")
            .with_property("name", serde_json::json!("Carol Davis"))
            .with_property("department", serde_json::json!("Marketing")),
        GraphNode::new()
            .with_label("Person")
            .with_label("Employee")
            .with_property("name", serde_json::json!("David Wilson"))
            .with_property("department", serde_json::json!("Sales")),
    ];

    let batch_results = store
        .create_nodes_batch(TEST_GRAPH_ID, additional_employees)
        .await?;
    let successful_creates = batch_results.iter().filter(|r| r.is_ok()).count();
    println!("✅ Batch created {successful_creates} additional employees");

    Ok(())
}

#[cfg(feature = "mock")]
async fn basic_relationship_operations(store: &MockGraphStore) -> TylResult<()> {
    println!("\n🔗 Basic Relationship Operations");
    println!("-------------------------------");

    // Find our previously created nodes
    let people = store
        .find_nodes(vec!["Person".to_string()], HashMap::new())
        .await?;
    let companies = store
        .find_nodes(vec!["Company".to_string()], HashMap::new())
        .await?;

    if people.len() >= 2 && !companies.is_empty() {
        let alice = &people[0];
        let bob = &people[1];
        let company = &companies[0];

        // Create various types of relationships
        let work_rel1 = GraphRelationship::new(&alice.id, &company.id, "WORKS_FOR")
            .with_property("start_date", serde_json::json!("2020-01-15"))
            .with_property("position", serde_json::json!("Senior Developer"))
            .with_property("salary_band", serde_json::json!("L4"));

        let work_rel2 = GraphRelationship::new(&bob.id, &company.id, "WORKS_FOR")
            .with_property("start_date", serde_json::json!("2021-03-01"))
            .with_property("position", serde_json::json!("UI/UX Designer"))
            .with_property("salary_band", serde_json::json!("L3"));

        let colleague_rel = GraphRelationship::new(&alice.id, &bob.id, "COLLEAGUE")
            .with_property("collaboration_projects", serde_json::json!(["Project X", "Project Y"]))
            .with_property("team", serde_json::json!("Product Team"));

        // Create the relationships
        let work1_id = store.create_relationship(TEST_GRAPH_ID, work_rel1).await?;
        let work2_id = store.create_relationship(TEST_GRAPH_ID, work_rel2).await?;
        let colleague_id = store
            .create_relationship(TEST_GRAPH_ID, colleague_rel)
            .await?;

        println!("✅ Created relationships:");
        println!("   - Alice WORKS_FOR TechCorp: {work1_id}");
        println!("   - Bob WORKS_FOR TechCorp: {work2_id}");
        println!("   - Alice is COLLEAGUE of Bob: {colleague_id}");

        // Update a relationship
        let mut rel_updates = HashMap::new();
        rel_updates.insert("performance_rating".to_string(), serde_json::json!("Excellent"));
        store
            .update_relationship(TEST_GRAPH_ID, &work1_id, rel_updates)
            .await?;

        println!("✅ Updated Alice's work relationship with performance rating");

        // Query relationships
        let work_relationships = store
            .find_relationships(vec!["WORKS_FOR".to_string()], HashMap::new())
            .await?;

        let rel_count = work_relationships.len();
        println!("📊 Found {rel_count} WORKS_FOR relationships");

        for rel in &work_relationships {
            if let Some(position) = rel.properties.get("position") {
                println!("   - Position: {position}");
            }
        }
    }

    Ok(())
}

#[cfg(feature = "mock")]
async fn graph_traversal_examples(store: &MockGraphStore) -> TylResult<()> {
    println!("\n🗺️  Graph Traversal Examples");
    println!("----------------------------");

    let people = store
        .find_nodes(vec!["Person".to_string()], HashMap::new())
        .await?;

    if !people.is_empty() {
        let alice = &people[0];

        // Find Alice's neighbors
        let neighbors = store
            .get_neighbors(&alice.id, TraversalParams::default())
            .await?;
        let neighbor_count = neighbors.len();
        println!("👥 Alice's neighbors ({neighbor_count} found):");

        for (neighbor, relationship) in &neighbors {
            println!(
                "   - {} via {} relationship",
                neighbor
                    .properties
                    .get("name")
                    .unwrap_or(&serde_json::json!("Unknown")),
                relationship.relationship_type
            );
        }

        // Traverse with filters
        let work_params = TraversalParams::new()
            .with_relationship_type("WORKS_FOR")
            .with_max_depth(2);

        let work_neighbors = store
            .get_neighbors(TEST_GRAPH_ID, &alice.id, work_params)
            .await?;
        let work_count = work_neighbors.len();
        println!("🏢 Alice's work-related connections: {work_count}");

        // Find path between two people if we have multiple
        if people.len() >= 2 {
            let bob = &people[1];
            let path = store
                .find_shortest_path(&alice.id, &bob.id, TraversalParams::default())
                .await?;

            if let Some(path) = path {
                println!("🛤️  Path from Alice to Bob:");
                let length = path.length;
                let weight = path.weight;
                let node_count = path.nodes.len();
                println!("   - Length: {length} relationships");
                println!("   - Weight: {weight:?}");
                println!("   - Nodes in path: {node_count}");

                for (i, node) in path.nodes.iter().enumerate() {
                    println!(
                        "     {}. {}",
                        i + 1,
                        node.properties
                            .get("name")
                            .unwrap_or(&serde_json::json!("Unknown"))
                    );
                }
            } else {
                println!("🚫 No path found between Alice and Bob");
            }
        }

        // Explore reachable nodes
        let reachable_params = TraversalParams::new().with_max_depth(3).with_limit(10);

        let reachable = store
            .traverse_from(TEST_GRAPH_ID, &alice.id, reachable_params)
            .await?;
        let reachable_count = reachable.len();
        println!("🌐 Nodes reachable from Alice (depth 3): {reachable_count}");

        for node in reachable.iter().take(5) {
            println!(
                "   - {} ({})",
                node.properties
                    .get("name")
                    .unwrap_or(&serde_json::json!("Unknown")),
                node.labels.join(", ")
            );
        }
    }

    Ok(())
}

#[cfg(feature = "mock")]
async fn graph_analytics_examples(store: &MockGraphStore) -> TylResult<()> {
    println!("\n📈 Graph Analytics Examples");
    println!("--------------------------");

    let people = store
        .find_nodes(vec!["Person".to_string()], HashMap::new())
        .await?;

    if !people.is_empty() {
        // Calculate degree centrality for all people
        let person_ids: Vec<String> = people.iter().map(|p| p.id.clone()).collect();
        let centrality = store
            .calculate_centrality(person_ids.clone(), CentralityType::Degree)
            .await?;

        println!("🎯 Degree Centrality Scores:");
        for (node_id, score) in &centrality {
            if let Some(person) = people.iter().find(|p| p.id == *node_id) {
                let default_name = serde_json::json!("Unknown");
                let name = person.properties.get("name").unwrap_or(&default_name);
                println!("   - {name}: {score:.1}");
            }
        }

        // Detect communities
        let communities = store
            .detect_communities(ClusteringAlgorithm::Louvain, HashMap::new())
            .await?;

        println!("🏘️  Community Detection:");
        let mut community_groups: HashMap<String, Vec<String>> = HashMap::new();

        for (node_id, community_id) in &communities {
            if let Some(person) = people.iter().find(|p| p.id == *node_id) {
                let name = person
                    .properties
                    .get("name")
                    .unwrap_or(&serde_json::json!("Unknown"))
                    .as_str()
                    .unwrap_or("Unknown")
                    .to_string();

                community_groups
                    .entry(community_id.clone())
                    .or_default()
                    .push(name);
            }
        }

        for (community_id, members) in community_groups {
            let member_list = members.join(", ");
            println!("   - {community_id}: {member_list}");
        }

        // Get relationship recommendations
        if let Some(alice) = people.first() {
            let recommendations = store
                .recommend_relationships(&alice.id, RecommendationType::CommonNeighbors, 3)
                .await?;

            println!(
                "💡 Relationship Recommendations for {}:",
                alice
                    .properties
                    .get("name")
                    .unwrap_or(&serde_json::json!("Alice"))
            );

            if recommendations.is_empty() {
                println!("   - No recommendations available (everyone already connected)");
            } else {
                for (target_id, rel_type, confidence) in recommendations {
                    println!("   - {rel_type} to {target_id} (confidence: {confidence:.2})");
                }
            }
        }

        // Find patterns (mock implementation returns empty, but shows the interface)
        let patterns = store.find_patterns(TEST_GRAPH_ID, 2, 1).await?;
        let pattern_count = patterns.len();
        println!("🔍 Graph Patterns (size 2+): {pattern_count} patterns found");
    }

    Ok(())
}

#[cfg(feature = "mock")]
async fn query_execution_examples(store: &MockGraphStore) -> TylResult<()> {
    println!("\n🔍 Query Execution Examples");
    println!("--------------------------");

    // Example read query
    let read_query = GraphQuery::read("SELECT name, department FROM persons ORDER BY name")
        .with_parameter("limit", serde_json::json!(10));

    let read_result = store.execute_read_query(TEST_GRAPH_ID, read_query).await?;
    println!("📊 Read query executed successfully");
    let data_count = read_result.data.len();
    let metadata_count = read_result.metadata.len();
    println!("   - Data rows: {data_count}");
    println!("   - Metadata entries: {metadata_count}");

    if let Some(execution_time) = read_result.metadata.get("execution_time_ms") {
        println!("   - Execution time: {execution_time}ms");
    }

    // Example write query
    let write_query = GraphQuery::write(
        "INSERT INTO temp_nodes (created_by, timestamp) VALUES ($creator, $time)",
    )
    .with_parameter("creator", serde_json::json!("basic_usage_example"))
    .with_parameter("time", serde_json::json!(chrono::Utc::now().to_rfc3339()));

    let write_result = store
        .execute_write_query(TEST_GRAPH_ID, write_query)
        .await?;
    println!("✏️  Write query executed successfully");
    let write_metadata_count = write_result.metadata.len();
    println!("   - Metadata entries: {write_metadata_count}");

    // Demonstrate query validation
    let invalid_write_as_read = GraphQuery::write("INSERT INTO test_nodes DEFAULT VALUES");
    match store
        .execute_read_query(TEST_GRAPH_ID, invalid_write_as_read)
        .await
    {
        Ok(_) => println!("❌ Unexpected success"),
        Err(_) => {
            println!("✅ Query validation working: write query rejected for read-only execution")
        }
    }

    Ok(())
}

#[cfg(feature = "mock")]
async fn health_monitoring_examples(store: &MockGraphStore) -> TylResult<()> {
    println!("\n🏥 Health Monitoring Examples");
    println!("----------------------------");

    // Basic health check
    let is_healthy = store.is_healthy().await?;
    println!(
        "❤️  System Health: {}",
        if is_healthy {
            "Healthy ✅"
        } else {
            "Unhealthy ❌"
        }
    );

    // Detailed health information
    let health_info = GraphHealth::health_check(store).await?;
    println!("📋 Detailed Health Information:");

    for (key, value) in &health_info {
        match key.as_str() {
            "status" => println!("   - Status: {value}"),
            "type" => println!("   - Store Type: {value}"),
            "uptime_seconds" => println!("   - Uptime: {value}s"),
            "version" => println!("   - Version: {value}"),
            _ => println!("   - {key}: {value}"),
        }
    }

    // System statistics
    let stats = store.get_all_statistics(TEST_GRAPH_ID).await?;
    println!("📊 System Statistics:");

    if let Some(node_count) = stats.get("node_count") {
        println!("   - Total Nodes: {node_count}");
    }

    if let Some(rel_count) = stats.get("relationship_count") {
        println!("   - Total Relationships: {rel_count}");
    }

    if let Some(labels) = stats.get("labels") {
        println!("   - Node Labels:");
        if let Ok(label_counts) = serde_json::from_value::<HashMap<String, usize>>(labels.clone()) {
            for (label, count) in label_counts {
                println!("     * {label}: {count} nodes");
            }
        }
    }

    if let Some(rel_types) = stats.get("relationship_types") {
        println!("   - Relationship Types:");
        if let Ok(type_counts) = serde_json::from_value::<HashMap<String, usize>>(rel_types.clone())
        {
            for (rel_type, count) in type_counts {
                println!("     * {rel_type}: {count} relationships");
            }
        }
    }

    Ok(())
}
