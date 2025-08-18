//! Mock implementation for testing and development
//!
//! This module provides a complete in-memory implementation of all graph traits,
//! suitable for testing, development, and scenarios where a full graph database
//! is not required.

#[cfg(feature = "mock")]
use crate::types::*;
#[cfg(feature = "mock")]
use crate::{
    graph_errors, GraphAnalytics, GraphHealth, GraphQueryExecutor, GraphStore, GraphTraversal,
};
#[cfg(feature = "mock")]
use async_trait::async_trait;
#[cfg(feature = "mock")]
use chrono::Utc;
#[cfg(feature = "mock")]
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(feature = "mock")]
use std::sync::Arc;
#[cfg(feature = "mock")]
use tyl_db_core::{DatabaseLifecycle, HealthCheckResult, HealthStatus};
#[cfg(feature = "mock")]
use tyl_errors::{TylError, TylResult};

#[cfg(feature = "mock")]
use tokio::sync::RwLock;

/// Mock graph store implementation using in-memory storage
///
/// This implementation provides full functionality for all graph traits
/// and is suitable for testing, development, and lightweight applications.
#[cfg(feature = "mock")]
#[derive(Debug, Default)]
pub struct MockGraphStore {
    nodes: Arc<RwLock<HashMap<String, GraphNode>>>,
    relationships: Arc<RwLock<HashMap<String, GraphRelationship>>>,
    next_id: Arc<RwLock<u64>>,
}

#[cfg(feature = "mock")]
impl MockGraphStore {
    /// Create a new mock graph store
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a unique ID for nodes/relationships
    async fn generate_id(&self) -> String {
        let mut counter = self.next_id.write().await;
        *counter += 1;
        format!("mock_{}", *counter)
    }

    /// Clear all data (useful for testing)
    pub async fn clear(&self) {
        let mut nodes = self.nodes.write().await;
        let mut relationships = self.relationships.write().await;
        nodes.clear();
        relationships.clear();

        let mut counter = self.next_id.write().await;
        *counter = 0;
    }

    /// Get node count (useful for testing)
    pub async fn node_count(&self) -> usize {
        let nodes = self.nodes.read().await;
        nodes.len()
    }

    /// Get relationship count (useful for testing)
    pub async fn relationship_count(&self) -> usize {
        let relationships = self.relationships.read().await;
        relationships.len()
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl DatabaseLifecycle for MockGraphStore {
    type Config = HashMap<String, serde_json::Value>;

    async fn connect(_config: Self::Config) -> TylResult<Self> {
        // Mock implementation - always succeeds
        Ok(Self::new())
    }

    async fn close(&mut self) -> TylResult<()> {
        // Mock implementation - clear data and always succeeds
        self.clear().await;
        Ok(())
    }

    fn connection_info(&self) -> String {
        "Mock Graph Store - In Memory".to_string()
    }

    async fn health_check(&self) -> TylResult<HealthCheckResult> {
        let health_info = <Self as GraphHealth>::health_check(self).await?;
        let start = std::time::Instant::now();
        // Simulate health check operation
        let response_time = start.elapsed();

        Ok(HealthCheckResult {
            status: HealthStatus::Healthy,
            timestamp: std::time::SystemTime::now(),
            response_time,
            metrics: health_info,
        })
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphStore for MockGraphStore {
    async fn create_node(&self, mut node: GraphNode) -> TylResult<String> {
        if node.id.is_empty() {
            node.id = self.generate_id().await;
        }

        let now = Utc::now();
        node.created_at = now;
        node.updated_at = now;

        let mut nodes = self.nodes.write().await;
        let id = node.id.clone();
        nodes.insert(id.clone(), node);
        Ok(id)
    }

    async fn create_nodes_batch(
        &self,
        nodes: Vec<GraphNode>,
    ) -> TylResult<Vec<Result<String, TylError>>> {
        let mut results = Vec::new();
        for node in nodes {
            match self.create_node(node).await {
                Ok(id) => results.push(Ok(id)),
                Err(e) => results.push(Err(e)),
            }
        }
        Ok(results)
    }

    async fn get_node(&self, id: &str) -> TylResult<Option<GraphNode>> {
        let nodes = self.nodes.read().await;
        Ok(nodes.get(id).cloned())
    }

    async fn update_node(
        &self,
        id: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<()> {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(id) {
            for (key, value) in properties {
                node.properties.insert(key, value);
            }
            node.updated_at = Utc::now();
            Ok(())
        } else {
            Err(graph_errors::node_not_found(id))
        }
    }

    async fn delete_node(&self, id: &str) -> TylResult<()> {
        let mut nodes = self.nodes.write().await;
        let mut relationships = self.relationships.write().await;

        // Remove the node
        nodes.remove(id);

        // Remove all relationships involving this node
        relationships.retain(|_, rel| rel.from_node_id != id && rel.to_node_id != id);

        Ok(())
    }

    async fn create_relationship(&self, mut relationship: GraphRelationship) -> TylResult<String> {
        if relationship.id.is_empty() {
            relationship.id = self.generate_id().await;
        }

        let now = Utc::now();
        relationship.created_at = now;
        relationship.updated_at = now;

        // Verify both nodes exist
        let nodes = self.nodes.read().await;
        if !nodes.contains_key(&relationship.from_node_id) {
            return Err(graph_errors::node_not_found(&relationship.from_node_id));
        }
        if !nodes.contains_key(&relationship.to_node_id) {
            return Err(graph_errors::node_not_found(&relationship.to_node_id));
        }
        drop(nodes);

        let mut relationships = self.relationships.write().await;
        let id = relationship.id.clone();
        relationships.insert(id.clone(), relationship);
        Ok(id)
    }

    async fn get_relationship(&self, id: &str) -> TylResult<Option<GraphRelationship>> {
        let relationships = self.relationships.read().await;
        Ok(relationships.get(id).cloned())
    }

    async fn update_relationship(
        &self,
        id: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<()> {
        let mut relationships = self.relationships.write().await;
        if let Some(relationship) = relationships.get_mut(id) {
            for (key, value) in properties {
                relationship.properties.insert(key, value);
            }
            relationship.updated_at = Utc::now();
            Ok(())
        } else {
            Err(graph_errors::relationship_not_found(id))
        }
    }

    async fn delete_relationship(&self, id: &str) -> TylResult<()> {
        let mut relationships = self.relationships.write().await;
        relationships.remove(id);
        Ok(())
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphTraversal for MockGraphStore {
    async fn get_neighbors(
        &self,
        node_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<(GraphNode, GraphRelationship)>> {
        let nodes = self.nodes.read().await;
        let relationships = self.relationships.read().await;

        if !nodes.contains_key(node_id) {
            return Err(graph_errors::node_not_found(node_id));
        }

        let mut neighbors = Vec::new();

        for relationship in relationships.values() {
            let connected_node_id = match params.direction {
                TraversalDirection::Outgoing => {
                    if relationship.from_node_id == node_id {
                        Some(&relationship.to_node_id)
                    } else {
                        None
                    }
                }
                TraversalDirection::Incoming => {
                    if relationship.to_node_id == node_id {
                        Some(&relationship.from_node_id)
                    } else {
                        None
                    }
                }
                TraversalDirection::Both => {
                    if relationship.from_node_id == node_id {
                        Some(&relationship.to_node_id)
                    } else if relationship.to_node_id == node_id {
                        Some(&relationship.from_node_id)
                    } else {
                        None
                    }
                }
            };

            if let Some(connected_id) = connected_node_id {
                if let Some(connected_node) = nodes.get(connected_id) {
                    // Apply filters
                    let relationship_matches = params.relationship_types.is_empty()
                        || params
                            .relationship_types
                            .contains(&relationship.relationship_type);
                    let node_matches = params.node_labels.is_empty()
                        || params
                            .node_labels
                            .iter()
                            .any(|label| connected_node.labels.contains(label));

                    // Apply property filters
                    let node_prop_matches = params
                        .node_filters
                        .iter()
                        .all(|(key, value)| connected_node.properties.get(key) == Some(value));
                    let rel_prop_matches = params
                        .relationship_filters
                        .iter()
                        .all(|(key, value)| relationship.properties.get(key) == Some(value));

                    if relationship_matches && node_matches && node_prop_matches && rel_prop_matches
                    {
                        neighbors.push((connected_node.clone(), relationship.clone()));
                    }
                }
            }
        }

        if let Some(limit) = params.limit {
            neighbors.truncate(limit);
        }

        Ok(neighbors)
    }

    async fn find_shortest_path(
        &self,
        from_id: &str,
        to_id: &str,
        params: TraversalParams,
    ) -> TylResult<Option<GraphPath>> {
        let nodes = self.nodes.read().await;
        let relationships = self.relationships.read().await;

        if !nodes.contains_key(from_id) || !nodes.contains_key(to_id) {
            return Ok(None);
        }

        if from_id == to_id {
            let node = nodes.get(from_id).unwrap().clone();
            return Ok(Some(GraphPath::new().add_node(node).with_weight(0.0)));
        }

        // Breadth-first search for shortest path
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent: HashMap<String, (String, String)> = HashMap::new(); // node_id -> (parent_id, relationship_id)

        queue.push_back(from_id.to_string());
        visited.insert(from_id.to_string());

        let max_depth = params.max_depth.unwrap_or(10);
        let mut depth = 0;

        while !queue.is_empty() && depth < max_depth {
            let level_size = queue.len();

            for _ in 0..level_size {
                let current = queue.pop_front().unwrap();

                if current == to_id {
                    // Reconstruct path
                    return Ok(Some(
                        self.reconstruct_path(&current, &parent, &nodes, &relationships, from_id)
                            .await,
                    ));
                }

                // Get neighbors
                for rel in relationships.values() {
                    let connected_node = self.get_connected_node(&current, rel, &params.direction);

                    if let Some(next_node) = connected_node {
                        if !visited.contains(next_node)
                            && self.matches_filters(rel, &nodes, next_node, &params).await
                        {
                            visited.insert(next_node.clone());
                            parent.insert(next_node.clone(), (current.clone(), rel.id.clone()));
                            queue.push_back(next_node.clone());
                        }
                    }
                }
            }

            depth += 1;
        }

        Ok(None)
    }

    async fn find_all_paths(
        &self,
        from_id: &str,
        to_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<GraphPath>> {
        let nodes = self.nodes.read().await;
        let relationships = self.relationships.read().await;

        if !nodes.contains_key(from_id) || !nodes.contains_key(to_id) {
            return Ok(Vec::new());
        }

        if from_id == to_id {
            let node = nodes.get(from_id).unwrap().clone();
            return Ok(vec![GraphPath::new().add_node(node).with_weight(0.0)]);
        }

        let max_depth = params.max_depth.unwrap_or(3);
        let mut all_paths = Vec::new();

        self.find_paths_recursive(
            from_id,
            to_id,
            vec![from_id.to_string()],
            Vec::new(),
            &mut HashSet::new(),
            0,
            max_depth,
            &nodes,
            &relationships,
            &mut all_paths,
            &params,
        )
        .await;

        // Sort by length/weight
        all_paths.sort_by(|a, b| a.length.cmp(&b.length));

        if let Some(limit) = params.limit {
            all_paths.truncate(limit);
        }

        Ok(all_paths)
    }

    async fn traverse_from(
        &self,
        start_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<GraphNode>> {
        let mut visited = Vec::new();
        let mut current_level = vec![start_id.to_string()];
        let mut depth = 0;

        // Add starting node
        if let Some(start_node) = self.get_node(start_id).await? {
            visited.push(start_node);
        }

        while !current_level.is_empty()
            && params.max_depth.is_none_or(|max| depth < max)
            && visited.len() < params.limit.unwrap_or(100)
        {
            let mut next_level = Vec::new();

            for node_id in current_level {
                if let Ok(neighbors) = self.get_neighbors(&node_id, params.clone()).await {
                    for (neighbor_node, _) in neighbors {
                        if !visited.iter().any(|n: &GraphNode| n.id == neighbor_node.id) {
                            visited.push(neighbor_node.clone());
                            next_level.push(neighbor_node.id);
                        }
                    }
                }
            }

            current_level = next_level;
            depth += 1;
        }

        Ok(visited)
    }

    async fn find_nodes(
        &self,
        labels: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<Vec<GraphNode>> {
        let nodes = self.nodes.read().await;

        let matching_nodes: Vec<GraphNode> = nodes
            .values()
            .filter(|node| {
                // Check labels
                let label_match =
                    labels.is_empty() || labels.iter().any(|label| node.labels.contains(label));

                // Check properties
                let property_match = properties
                    .iter()
                    .all(|(key, value)| node.properties.get(key) == Some(value));

                label_match && property_match
            })
            .cloned()
            .collect();

        Ok(matching_nodes)
    }

    async fn find_relationships(
        &self,
        relationship_types: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<Vec<GraphRelationship>> {
        let relationships = self.relationships.read().await;

        let matching_relationships: Vec<GraphRelationship> = relationships
            .values()
            .filter(|rel| {
                // Check relationship types
                let type_match = relationship_types.is_empty()
                    || relationship_types.contains(&rel.relationship_type);

                // Check properties
                let property_match = properties
                    .iter()
                    .all(|(key, value)| rel.properties.get(key) == Some(value));

                type_match && property_match
            })
            .cloned()
            .collect();

        Ok(matching_relationships)
    }
}

// Helper methods for MockGraphStore
#[cfg(feature = "mock")]
impl MockGraphStore {
    async fn reconstruct_path(
        &self,
        target: &str,
        parent: &HashMap<String, (String, String)>,
        nodes: &HashMap<String, GraphNode>,
        relationships: &HashMap<String, GraphRelationship>,
        start: &str,
    ) -> GraphPath {
        let mut path_nodes = Vec::new();
        let mut path_relationships = Vec::new();
        let mut current = target.to_string();

        // Build path backwards
        let mut path_ids = Vec::new();
        while let Some((parent_id, rel_id)) = parent.get(&current) {
            path_ids.push((parent_id.clone(), rel_id.clone(), current.clone()));
            current = parent_id.clone();
        }
        path_ids.reverse();

        // Add starting node
        if let Some(start_node) = nodes.get(start) {
            path_nodes.push(start_node.clone());
        }

        // Add intermediate nodes and relationships
        for (_, rel_id, target_id) in path_ids {
            if let Some(rel) = relationships.get(&rel_id) {
                path_relationships.push(rel.clone());
            }
            if let Some(node) = nodes.get(&target_id) {
                path_nodes.push(node.clone());
            }
        }

        let path_length = path_relationships.len();
        GraphPath {
            nodes: path_nodes,
            relationships: path_relationships,
            length: path_length,
            weight: Some(path_length as f64),
        }
    }

    fn get_connected_node<'a>(
        &self,
        current_id: &str,
        relationship: &'a GraphRelationship,
        direction: &TraversalDirection,
    ) -> Option<&'a String> {
        match direction {
            TraversalDirection::Outgoing => {
                if relationship.from_node_id == current_id {
                    Some(&relationship.to_node_id)
                } else {
                    None
                }
            }
            TraversalDirection::Incoming => {
                if relationship.to_node_id == current_id {
                    Some(&relationship.from_node_id)
                } else {
                    None
                }
            }
            TraversalDirection::Both => {
                if relationship.from_node_id == current_id {
                    Some(&relationship.to_node_id)
                } else if relationship.to_node_id == current_id {
                    Some(&relationship.from_node_id)
                } else {
                    None
                }
            }
        }
    }

    async fn matches_filters(
        &self,
        relationship: &GraphRelationship,
        nodes: &HashMap<String, GraphNode>,
        node_id: &str,
        params: &TraversalParams,
    ) -> bool {
        // Check relationship type
        let relationship_matches = params.relationship_types.is_empty()
            || params
                .relationship_types
                .contains(&relationship.relationship_type);

        // Check node labels
        let node_matches = if let Some(node) = nodes.get(node_id) {
            params.node_labels.is_empty()
                || params
                    .node_labels
                    .iter()
                    .any(|label| node.labels.contains(label))
        } else {
            false
        };

        // Check property filters
        let node_prop_matches = if let Some(node) = nodes.get(node_id) {
            params
                .node_filters
                .iter()
                .all(|(key, value)| node.properties.get(key) == Some(value))
        } else {
            false
        };

        let rel_prop_matches = params
            .relationship_filters
            .iter()
            .all(|(key, value)| relationship.properties.get(key) == Some(value));

        relationship_matches && node_matches && node_prop_matches && rel_prop_matches
    }

    #[allow(clippy::too_many_arguments)]
    async fn find_paths_recursive(
        &self,
        current_id: &str,
        target_id: &str,
        current_path: Vec<String>,
        current_relationships: Vec<String>,
        visited: &mut HashSet<String>,
        depth: usize,
        max_depth: usize,
        nodes: &HashMap<String, GraphNode>,
        relationships: &HashMap<String, GraphRelationship>,
        all_paths: &mut Vec<GraphPath>,
        params: &TraversalParams,
    ) {
        if depth >= max_depth {
            return;
        }

        if current_id == target_id && !current_path.is_empty() && current_path.len() > 1 {
            // Found a path, construct GraphPath
            let mut path_nodes = Vec::new();
            let mut path_rels = Vec::new();

            for node_id in &current_path {
                if let Some(node) = nodes.get(node_id) {
                    path_nodes.push(node.clone());
                }
            }

            for rel_id in &current_relationships {
                if let Some(rel) = relationships.get(rel_id) {
                    path_rels.push(rel.clone());
                }
            }

            all_paths.push(GraphPath {
                nodes: path_nodes,
                relationships: path_rels,
                length: current_relationships.len(),
                weight: Some(current_relationships.len() as f64),
            });
            return;
        }

        visited.insert(current_id.to_string());

        for rel in relationships.values() {
            let next_node = self.get_connected_node(current_id, rel, &params.direction);

            if let Some(next_id) = next_node {
                if !visited.contains(next_id)
                    && self.matches_filters(rel, nodes, next_id, params).await
                {
                    let mut new_path = current_path.clone();
                    new_path.push(next_id.clone());

                    let mut new_relationships = current_relationships.clone();
                    new_relationships.push(rel.id.clone());

                    Box::pin(self.find_paths_recursive(
                        next_id,
                        target_id,
                        new_path,
                        new_relationships,
                        visited,
                        depth + 1,
                        max_depth,
                        nodes,
                        relationships,
                        all_paths,
                        params,
                    ))
                    .await;
                }
            }
        }

        visited.remove(current_id);
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphAnalytics for MockGraphStore {
    async fn calculate_centrality(
        &self,
        node_ids: Vec<String>,
        centrality_type: CentralityType,
    ) -> TylResult<HashMap<String, f64>> {
        let nodes = self.nodes.read().await;
        let relationships = self.relationships.read().await;

        let target_nodes = if node_ids.is_empty() {
            nodes.keys().cloned().collect()
        } else {
            node_ids
        };

        let mut centrality_scores = HashMap::new();

        match centrality_type {
            CentralityType::Degree => {
                for node_id in target_nodes {
                    let degree = relationships
                        .values()
                        .filter(|rel| rel.from_node_id == node_id || rel.to_node_id == node_id)
                        .count();
                    centrality_scores.insert(node_id, degree as f64);
                }
            }
            CentralityType::Betweenness => {
                // Simplified betweenness centrality (for demo purposes)
                for node_id in target_nodes {
                    centrality_scores.insert(node_id, 0.5); // Mock value
                }
            }
            CentralityType::Closeness => {
                // Simplified closeness centrality (for demo purposes)
                for node_id in target_nodes {
                    centrality_scores.insert(node_id, 0.7); // Mock value
                }
            }
            CentralityType::Eigenvector => {
                // Simplified eigenvector centrality (for demo purposes)
                for node_id in target_nodes {
                    centrality_scores.insert(node_id, 0.6); // Mock value
                }
            }
            CentralityType::PageRank => {
                // Simplified PageRank (for demo purposes)
                for node_id in target_nodes {
                    centrality_scores.insert(node_id, 0.8); // Mock value
                }
            }
        }

        Ok(centrality_scores)
    }

    async fn detect_communities(
        &self,
        _algorithm: ClusteringAlgorithm,
        _params: HashMap<String, serde_json::Value>,
    ) -> TylResult<HashMap<String, String>> {
        let nodes = self.nodes.read().await;
        let mut communities = HashMap::new();

        // Simple mock implementation - assign nodes to communities based on labels
        for (node_id, node) in nodes.iter() {
            let community_id = if node.labels.is_empty() {
                "default_community".to_string()
            } else {
                format!("community_{}", node.labels[0])
            };
            communities.insert(node_id.clone(), community_id);
        }

        Ok(communities)
    }

    async fn find_patterns(
        &self,
        _pattern_size: usize,
        _min_frequency: usize,
    ) -> TylResult<Vec<(GraphPath, usize)>> {
        // Mock implementation - return empty patterns
        Ok(Vec::new())
    }

    async fn recommend_relationships(
        &self,
        node_id: &str,
        _recommendation_type: RecommendationType,
        limit: usize,
    ) -> TylResult<Vec<(String, String, f64)>> {
        let nodes = self.nodes.read().await;
        let relationships = self.relationships.read().await;

        if !nodes.contains_key(node_id) {
            return Err(graph_errors::node_not_found(node_id));
        }

        // Simple mock recommendation - suggest connections to nodes not already connected
        let connected_nodes: HashSet<String> = relationships
            .values()
            .filter_map(|rel| {
                if rel.from_node_id == node_id {
                    Some(rel.to_node_id.clone())
                } else if rel.to_node_id == node_id {
                    Some(rel.from_node_id.clone())
                } else {
                    None
                }
            })
            .collect();

        let recommendations: Vec<(String, String, f64)> = nodes
            .keys()
            .filter(|id| *id != node_id && !connected_nodes.contains(*id))
            .take(limit)
            .map(|target_id| (target_id.clone(), "RECOMMENDED".to_string(), 0.75))
            .collect();

        Ok(recommendations)
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphQueryExecutor for MockGraphStore {
    async fn execute_query(&self, _query: GraphQuery) -> TylResult<QueryResult> {
        // Mock implementation - return empty result
        Ok(QueryResult::new()
            .with_metadata("executed_at", serde_json::json!(Utc::now().to_rfc3339()))
            .with_metadata("execution_time_ms", serde_json::json!(1)))
    }

    async fn execute_read_query(&self, query: GraphQuery) -> TylResult<QueryResult> {
        if query.is_write_query {
            return Err(graph_errors::invalid_query("Cannot execute write query as read-only"));
        }
        self.execute_query(query).await
    }

    async fn execute_write_query(&self, query: GraphQuery) -> TylResult<QueryResult> {
        if !query.is_write_query {
            return Err(graph_errors::invalid_query("Cannot execute read query as write"));
        }
        self.execute_query(query).await
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphHealth for MockGraphStore {
    async fn is_healthy(&self) -> TylResult<bool> {
        Ok(true)
    }

    async fn health_check(&self) -> TylResult<HashMap<String, serde_json::Value>> {
        let mut health = HashMap::new();
        health.insert("status".to_string(), serde_json::json!("healthy"));
        health.insert("type".to_string(), serde_json::json!("mock"));
        health.insert("uptime_seconds".to_string(), serde_json::json!(3600)); // Mock uptime
        health.insert("version".to_string(), serde_json::json!("1.0.0"));
        Ok(health)
    }

    async fn get_statistics(&self) -> TylResult<HashMap<String, serde_json::Value>> {
        let nodes = self.nodes.read().await;
        let relationships = self.relationships.read().await;

        let mut stats = HashMap::new();
        stats.insert("node_count".to_string(), serde_json::json!(nodes.len()));
        stats.insert("relationship_count".to_string(), serde_json::json!(relationships.len()));
        stats.insert("labels".to_string(), {
            let mut label_counts: HashMap<String, usize> = HashMap::new();
            for node in nodes.values() {
                for label in &node.labels {
                    *label_counts.entry(label.clone()).or_insert(0) += 1;
                }
            }
            serde_json::to_value(label_counts).unwrap()
        });
        stats.insert("relationship_types".to_string(), {
            let mut type_counts: HashMap<String, usize> = HashMap::new();
            for rel in relationships.values() {
                *type_counts
                    .entry(rel.relationship_type.clone())
                    .or_insert(0) += 1;
            }
            serde_json::to_value(type_counts).unwrap()
        });

        Ok(stats)
    }
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_graph_store_node_operations_should_work() {
        // TDD: Basic CRUD operations for nodes
        let store = MockGraphStore::new();

        // Create node
        let node = GraphNode::new()
            .with_label("Person")
            .with_property("name", serde_json::json!("Alice"));

        let node_id = store.create_node(node.clone()).await.unwrap();
        assert!(!node_id.is_empty());

        // Get node
        let retrieved = store.get_node(&node_id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved_node = retrieved.unwrap();
        assert_eq!(retrieved_node.labels, vec!["Person"]);
        assert_eq!(retrieved_node.properties.get("name"), Some(&serde_json::json!("Alice")));

        // Update node
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), serde_json::json!(30));
        store.update_node(&node_id, updates).await.unwrap();

        let updated = store.get_node(&node_id).await.unwrap().unwrap();
        assert_eq!(updated.properties.get("age"), Some(&serde_json::json!(30)));

        // Delete node
        store.delete_node(&node_id).await.unwrap();
        let deleted = store.get_node(&node_id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_mock_graph_store_relationship_operations_should_work() {
        // TDD: Basic CRUD operations for relationships
        let store = MockGraphStore::new();

        // Create nodes first
        let node1 = GraphNode::new().with_label("Person");
        let node2 = GraphNode::new().with_label("Person");
        let node1_id = store.create_node(node1).await.unwrap();
        let node2_id = store.create_node(node2).await.unwrap();

        // Create relationship
        let rel = GraphRelationship::new(&node1_id, &node2_id, "KNOWS")
            .with_property("since", serde_json::json!("2020"));

        let rel_id = store.create_relationship(rel).await.unwrap();
        assert!(!rel_id.is_empty());

        // Get relationship
        let retrieved = store.get_relationship(&rel_id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved_rel = retrieved.unwrap();
        assert_eq!(retrieved_rel.relationship_type, "KNOWS");
        assert_eq!(retrieved_rel.from_node_id, node1_id);
        assert_eq!(retrieved_rel.to_node_id, node2_id);

        // Update relationship
        let mut updates = HashMap::new();
        updates.insert("strength".to_string(), serde_json::json!(0.8));
        store.update_relationship(&rel_id, updates).await.unwrap();

        let updated = store.get_relationship(&rel_id).await.unwrap().unwrap();
        assert_eq!(updated.properties.get("strength"), Some(&serde_json::json!(0.8)));

        // Delete relationship
        store.delete_relationship(&rel_id).await.unwrap();
        let deleted = store.get_relationship(&rel_id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_mock_graph_store_traversal_should_work() {
        // TDD: Graph traversal functionality
        let store = MockGraphStore::new();

        // Create a simple graph: A -> B -> C
        let node_a = GraphNode::with_id("A").with_label("Person");
        let node_b = GraphNode::with_id("B").with_label("Person");
        let node_c = GraphNode::with_id("C").with_label("Person");

        store.create_node(node_a).await.unwrap();
        store.create_node(node_b).await.unwrap();
        store.create_node(node_c).await.unwrap();

        let rel_ab = GraphRelationship::new("A", "B", "KNOWS");
        let rel_bc = GraphRelationship::new("B", "C", "KNOWS");

        store.create_relationship(rel_ab).await.unwrap();
        store.create_relationship(rel_bc).await.unwrap();

        // Test neighbors
        let neighbors = store
            .get_neighbors("A", TraversalParams::default())
            .await
            .unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0.id, "B");

        // Test shortest path
        let path = store
            .find_shortest_path("A", "C", TraversalParams::default())
            .await
            .unwrap();
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.length, 2);
        assert_eq!(path.nodes.len(), 3);
        assert_eq!(path.nodes[0].id, "A");
        assert_eq!(path.nodes[2].id, "C");

        // Test traversal
        let reachable = store
            .traverse_from("A", TraversalParams::default())
            .await
            .unwrap();
        assert_eq!(reachable.len(), 3); // A, B, C
    }

    #[tokio::test]
    async fn test_mock_graph_store_health_should_work() {
        // TDD: Health checking functionality
        let store = MockGraphStore::new();

        // Test health check
        let is_healthy = store.is_healthy().await.unwrap();
        assert!(is_healthy);

        let health = GraphHealth::health_check(&store).await.unwrap();
        assert_eq!(health.get("status"), Some(&serde_json::json!("healthy")));
        assert_eq!(health.get("type"), Some(&serde_json::json!("mock")));

        // Test statistics
        let stats = store.get_statistics().await.unwrap();
        assert_eq!(stats.get("node_count"), Some(&serde_json::json!(0)));
        assert_eq!(stats.get("relationship_count"), Some(&serde_json::json!(0)));
    }

    #[tokio::test]
    async fn test_mock_graph_store_analytics_should_work() {
        // TDD: Graph analytics functionality
        let store = MockGraphStore::new();

        // Create some test data
        let node_a = GraphNode::with_id("A").with_label("Person");
        let node_b = GraphNode::with_id("B").with_label("Person");
        store.create_node(node_a).await.unwrap();
        store.create_node(node_b).await.unwrap();

        let rel = GraphRelationship::new("A", "B", "KNOWS");
        store.create_relationship(rel).await.unwrap();

        // Test centrality calculation
        let centrality = store
            .calculate_centrality(vec!["A".to_string()], CentralityType::Degree)
            .await
            .unwrap();
        assert!(centrality.contains_key("A"));
        assert_eq!(centrality.get("A"), Some(&1.0)); // A has degree 1

        // Test community detection
        let communities = store
            .detect_communities(ClusteringAlgorithm::Louvain, HashMap::new())
            .await
            .unwrap();
        assert!(communities.contains_key("A"));
        assert!(communities.contains_key("B"));

        // Test recommendations
        let recommendations = store
            .recommend_relationships("A", RecommendationType::SimilarNodes, 10)
            .await
            .unwrap();
        assert!(recommendations.is_empty()); // B is already connected, so no recommendations
    }
}
