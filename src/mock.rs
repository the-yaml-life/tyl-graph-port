//! Complete Mock implementation for testing and development with full multi-graph support
//!
//! This module provides a complete in-memory implementation of all graph traits,
//! suitable for testing, development, and scenarios where a full graph database
//! is not required.

#[cfg(feature = "mock")]
use crate::types::*;
#[cfg(feature = "mock")]
use crate::{
    graph_errors, GraphAnalytics, GraphBulkOperations, GraphConstraintManager, GraphHealth,
    GraphIndexManager, GraphQueryExecutor, GraphStore, GraphTransaction, GraphTraversal,
    MultiGraphManager,
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

/// Graph data container for multi-graph support
#[cfg(feature = "mock")]
#[derive(Debug, Default)]
struct GraphData {
    /// Nodes in this graph
    nodes: HashMap<String, GraphNode>,
    /// Relationships in this graph
    relationships: HashMap<String, GraphRelationship>,
    /// Graph metadata
    info: GraphInfo,
}

#[cfg(feature = "mock")]
impl GraphData {
    fn new(info: GraphInfo) -> Self {
        Self { nodes: HashMap::new(), relationships: HashMap::new(), info }
    }
}

/// Mock graph store implementation using in-memory storage with multi-graph support
///
/// This implementation provides full functionality for all graph traits
/// and is suitable for testing, development, and lightweight applications.
#[cfg(feature = "mock")]
#[derive(Debug, Default)]
pub struct MockGraphStore {
    /// Multi-graph storage: graph_id -> GraphData
    graphs: Arc<RwLock<HashMap<String, GraphData>>>,
    /// Global counter for generating unique IDs
    next_id: Arc<RwLock<u64>>,
    /// Active transactions
    transactions: Arc<RwLock<HashMap<String, TransactionContext>>>,
    /// Indexes per graph
    indexes: Arc<RwLock<HashMap<String, Vec<IndexConfig>>>>,
    /// Constraints per graph
    constraints: Arc<RwLock<HashMap<String, Vec<ConstraintConfig>>>>,
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
        let mut graphs = self.graphs.write().await;
        let mut transactions = self.transactions.write().await;
        let mut indexes = self.indexes.write().await;
        let mut constraints = self.constraints.write().await;

        graphs.clear();
        transactions.clear();
        indexes.clear();
        constraints.clear();

        let mut counter = self.next_id.write().await;
        *counter = 0;
    }

    /// Get node count for a specific graph (useful for testing)
    pub async fn node_count(&self, graph_id: &str) -> usize {
        let graphs = self.graphs.read().await;
        graphs
            .get(graph_id)
            .map(|graph| graph.nodes.len())
            .unwrap_or(0)
    }

    /// Get relationship count for a specific graph (useful for testing)
    pub async fn relationship_count(&self, graph_id: &str) -> usize {
        let graphs = self.graphs.read().await;
        graphs
            .get(graph_id)
            .map(|graph| graph.relationships.len())
            .unwrap_or(0)
    }

    /// Get total node count across all graphs
    pub async fn total_node_count(&self) -> usize {
        let graphs = self.graphs.read().await;
        graphs.values().map(|graph| graph.nodes.len()).sum()
    }

    /// Get total relationship count across all graphs
    pub async fn total_relationship_count(&self) -> usize {
        let graphs = self.graphs.read().await;
        graphs.values().map(|graph| graph.relationships.len()).sum()
    }

    /// Ensure graph exists, return error if not
    async fn ensure_graph_exists(&self, graph_id: &str) -> TylResult<()> {
        let graphs = self.graphs.read().await;
        if graphs.contains_key(graph_id) {
            Ok(())
        } else {
            Err(TylError::not_found("graph", graph_id))
        }
    }
}

// ===== DATABASE LIFECYCLE IMPLEMENTATION =====

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

// ===== MULTI-GRAPH MANAGER IMPLEMENTATION =====

#[cfg(feature = "mock")]
#[async_trait]
impl MultiGraphManager for MockGraphStore {
    async fn create_graph(&self, graph_info: GraphInfo) -> TylResult<GraphInfo> {
        let mut graphs = self.graphs.write().await;

        if graphs.contains_key(&graph_info.id) {
            return Err(TylError::validation(
                "graph_id",
                format!("Graph '{}' already exists", graph_info.id),
            ));
        }

        let graph_data = GraphData::new(graph_info.clone());
        graphs.insert(graph_info.id.clone(), graph_data);

        Ok(graph_info)
    }

    async fn list_graphs(&self) -> TylResult<Vec<GraphInfo>> {
        let graphs = self.graphs.read().await;
        let graph_infos = graphs.values().map(|graph| graph.info.clone()).collect();
        Ok(graph_infos)
    }

    async fn get_graph(&self, graph_id: &str) -> TylResult<Option<GraphInfo>> {
        let graphs = self.graphs.read().await;
        Ok(graphs.get(graph_id).map(|graph| graph.info.clone()))
    }

    async fn update_graph_metadata(
        &self,
        graph_id: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> TylResult<()> {
        let mut graphs = self.graphs.write().await;

        if let Some(graph) = graphs.get_mut(graph_id) {
            for (key, value) in metadata {
                graph.info.metadata.insert(key, value);
            }
            graph.info.updated_at = Utc::now();
            Ok(())
        } else {
            Err(TylError::not_found("graph", graph_id))
        }
    }

    async fn delete_graph(&self, graph_id: &str) -> TylResult<()> {
        let mut graphs = self.graphs.write().await;
        let mut indexes = self.indexes.write().await;
        let mut constraints = self.constraints.write().await;
        let mut transactions = self.transactions.write().await;

        graphs.remove(graph_id);
        indexes.remove(graph_id);
        constraints.remove(graph_id);

        // Remove any active transactions for this graph
        transactions.retain(|_, tx| {
            !tx.metadata
                .get("graph_id")
                .map(|v| v.as_str().unwrap_or(""))
                .eq(&Some(graph_id))
        });

        Ok(())
    }

    async fn graph_exists(&self, graph_id: &str) -> TylResult<bool> {
        let graphs = self.graphs.read().await;
        Ok(graphs.contains_key(graph_id))
    }
}

// ===== GRAPH STORE IMPLEMENTATION =====

#[cfg(feature = "mock")]
#[async_trait]
impl GraphStore for MockGraphStore {
    async fn create_node(&self, graph_id: &str, mut node: GraphNode) -> TylResult<String> {
        self.ensure_graph_exists(graph_id).await?;

        if node.id.is_empty() {
            node.id = self.generate_id().await;
        }

        let now = Utc::now();
        node.created_at = now;
        node.updated_at = now;

        let mut graphs = self.graphs.write().await;
        let graph = graphs.get_mut(graph_id).unwrap(); // We already checked existence
        let id = node.id.clone();
        graph.nodes.insert(id.clone(), node);
        Ok(id)
    }

    async fn create_nodes_batch(
        &self,
        graph_id: &str,
        nodes: Vec<GraphNode>,
    ) -> TylResult<Vec<Result<String, TylError>>> {
        let mut results = Vec::new();
        for node in nodes {
            match self.create_node(graph_id, node).await {
                Ok(id) => results.push(Ok(id)),
                Err(e) => results.push(Err(e)),
            }
        }
        Ok(results)
    }

    async fn get_node(&self, graph_id: &str, id: &str) -> TylResult<Option<GraphNode>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap(); // We already checked existence
        Ok(graph.nodes.get(id).cloned())
    }

    async fn update_node(
        &self,
        graph_id: &str,
        id: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut graphs = self.graphs.write().await;
        let graph = graphs.get_mut(graph_id).unwrap(); // We already checked existence

        if let Some(node) = graph.nodes.get_mut(id) {
            for (key, value) in properties {
                node.properties.insert(key, value);
            }
            node.updated_at = Utc::now();
            Ok(())
        } else {
            Err(graph_errors::node_not_found(id))
        }
    }

    async fn delete_node(&self, graph_id: &str, id: &str) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut graphs = self.graphs.write().await;
        let graph = graphs.get_mut(graph_id).unwrap(); // We already checked existence

        // Remove the node
        graph.nodes.remove(id);

        // Remove all relationships involving this node
        graph
            .relationships
            .retain(|_, rel| rel.from_node_id != id && rel.to_node_id != id);

        Ok(())
    }

    async fn create_relationship(
        &self,
        graph_id: &str,
        mut relationship: GraphRelationship,
    ) -> TylResult<String> {
        self.ensure_graph_exists(graph_id).await?;

        if relationship.id.is_empty() {
            relationship.id = self.generate_id().await;
        }

        let now = Utc::now();
        relationship.created_at = now;
        relationship.updated_at = now;

        let mut graphs = self.graphs.write().await;
        let graph = graphs.get_mut(graph_id).unwrap(); // We already checked existence

        // Verify both nodes exist in this graph
        if !graph.nodes.contains_key(&relationship.from_node_id) {
            return Err(graph_errors::node_not_found(&relationship.from_node_id));
        }
        if !graph.nodes.contains_key(&relationship.to_node_id) {
            return Err(graph_errors::node_not_found(&relationship.to_node_id));
        }

        let id = relationship.id.clone();
        graph.relationships.insert(id.clone(), relationship);
        Ok(id)
    }

    async fn get_relationship(
        &self,
        graph_id: &str,
        id: &str,
    ) -> TylResult<Option<GraphRelationship>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap(); // We already checked existence
        Ok(graph.relationships.get(id).cloned())
    }

    async fn update_relationship(
        &self,
        graph_id: &str,
        id: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut graphs = self.graphs.write().await;
        let graph = graphs.get_mut(graph_id).unwrap(); // We already checked existence

        if let Some(relationship) = graph.relationships.get_mut(id) {
            for (key, value) in properties {
                relationship.properties.insert(key, value);
            }
            relationship.updated_at = Utc::now();
            Ok(())
        } else {
            Err(graph_errors::relationship_not_found(id))
        }
    }

    async fn delete_relationship(&self, graph_id: &str, id: &str) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut graphs = self.graphs.write().await;
        let graph = graphs.get_mut(graph_id).unwrap(); // We already checked existence
        graph.relationships.remove(id);
        Ok(())
    }
}

// ===== GRAPH TRAVERSAL IMPLEMENTATION =====

#[cfg(feature = "mock")]
#[async_trait]
impl GraphTraversal for MockGraphStore {
    async fn get_neighbors(
        &self,
        graph_id: &str,
        node_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<(GraphNode, GraphRelationship)>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        if !graph.nodes.contains_key(node_id) {
            return Err(graph_errors::node_not_found(node_id));
        }

        let mut neighbors = Vec::new();

        for (_, relationship) in graph.relationships.iter() {
            let is_match = match params.direction {
                TraversalDirection::Outgoing => relationship.from_node_id == node_id,
                TraversalDirection::Incoming => relationship.to_node_id == node_id,
                TraversalDirection::Both => {
                    relationship.from_node_id == node_id || relationship.to_node_id == node_id
                }
            };

            if !is_match {
                continue;
            }

            // Check relationship type filter
            if !params.relationship_types.is_empty()
                && !params
                    .relationship_types
                    .contains(&relationship.relationship_type)
            {
                continue;
            }

            // Check relationship property filters
            let mut relationship_matches = true;
            for (key, expected_value) in &params.relationship_filters {
                if let Some(actual_value) = relationship.properties.get(key) {
                    if actual_value != expected_value {
                        relationship_matches = false;
                        break;
                    }
                } else {
                    relationship_matches = false;
                    break;
                }
            }

            if !relationship_matches {
                continue;
            }

            // Get the neighbor node
            let neighbor_id = if relationship.from_node_id == node_id {
                &relationship.to_node_id
            } else {
                &relationship.from_node_id
            };

            if let Some(neighbor_node) = graph.nodes.get(neighbor_id) {
                // Check node label filter
                if !params.node_labels.is_empty() {
                    let has_matching_label = neighbor_node
                        .labels
                        .iter()
                        .any(|label| params.node_labels.contains(label));
                    if !has_matching_label {
                        continue;
                    }
                }

                // Check node property filters
                let mut node_matches = true;
                for (key, expected_value) in &params.node_filters {
                    if let Some(actual_value) = neighbor_node.properties.get(key) {
                        if actual_value != expected_value {
                            node_matches = false;
                            break;
                        }
                    } else {
                        node_matches = false;
                        break;
                    }
                }

                if node_matches {
                    neighbors.push((neighbor_node.clone(), relationship.clone()));
                }
            }
        }

        // Apply limit
        if let Some(limit) = params.limit {
            neighbors.truncate(limit);
        }

        Ok(neighbors)
    }

    async fn find_shortest_path(
        &self,
        graph_id: &str,
        from_id: &str,
        to_id: &str,
        params: TraversalParams,
    ) -> TylResult<Option<GraphPath>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        if !graph.nodes.contains_key(from_id) || !graph.nodes.contains_key(to_id) {
            return Ok(None);
        }

        if from_id == to_id {
            // Return path with just the starting node
            if let Some(node) = graph.nodes.get(from_id) {
                return Ok(Some(GraphPath {
                    nodes: vec![node.clone()],
                    relationships: vec![],
                    length: 0,
                    weight: Some(0.0),
                }));
            }
        }

        // Breadth-first search for shortest path
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent: HashMap<String, (String, String)> = HashMap::new(); // node_id -> (parent_id, relationship_id)

        queue.push_back(from_id.to_string());
        visited.insert(from_id.to_string());

        while let Some(current_node_id) = queue.pop_front() {
            if current_node_id == to_id {
                // Found path - reconstruct it
                let mut path_nodes = Vec::new();
                let mut path_relationships = Vec::new();

                let mut current = to_id.to_string();

                // Build path backwards
                let mut path_ids = Vec::new();
                while let Some((parent_id, rel_id)) = parent.get(&current) {
                    path_ids.push((parent_id.clone(), rel_id.clone(), current.clone()));
                    current = parent_id.clone();
                }
                path_ids.reverse();

                // Add starting node
                if let Some(start_node) = graph.nodes.get(from_id) {
                    path_nodes.push(start_node.clone());
                }

                // Add intermediate nodes and relationships
                for (_, rel_id, target_id) in path_ids {
                    if let Some(rel) = graph.relationships.get(&rel_id) {
                        path_relationships.push(rel.clone());
                    }
                    if let Some(node) = graph.nodes.get(&target_id) {
                        path_nodes.push(node.clone());
                    }
                }

                return Ok(Some(GraphPath {
                    nodes: path_nodes,
                    relationships: path_relationships.clone(),
                    length: path_relationships.len(),
                    weight: Some(path_relationships.len() as f64),
                }));
            }

            // Explore neighbors
            for (_, relationship) in graph.relationships.iter() {
                // Apply direction filter
                let next_node_id = match params.direction {
                    TraversalDirection::Outgoing
                        if relationship.from_node_id == current_node_id =>
                    {
                        Some(&relationship.to_node_id)
                    }
                    TraversalDirection::Incoming if relationship.to_node_id == current_node_id => {
                        Some(&relationship.from_node_id)
                    }
                    TraversalDirection::Both => {
                        if relationship.from_node_id == current_node_id {
                            Some(&relationship.to_node_id)
                        } else if relationship.to_node_id == current_node_id {
                            Some(&relationship.from_node_id)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(next_id) = next_node_id {
                    if !visited.contains(next_id) {
                        // Apply relationship type filter
                        if !params.relationship_types.is_empty()
                            && !params
                                .relationship_types
                                .contains(&relationship.relationship_type)
                        {
                            continue;
                        }

                        visited.insert(next_id.clone());
                        parent.insert(
                            next_id.clone(),
                            (current_node_id.clone(), relationship.id.clone()),
                        );
                        queue.push_back(next_id.clone());
                    }
                }
            }
        }

        Ok(None) // No path found
    }

    async fn find_shortest_weighted_path(
        &self,
        graph_id: &str,
        from_id: &str,
        to_id: &str,
        params: TraversalParams,
        weight_property: &str,
    ) -> TylResult<Option<WeightedPath>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        if !graph.nodes.contains_key(from_id) || !graph.nodes.contains_key(to_id) {
            return Ok(None);
        }

        // Simple implementation using Dijkstra's algorithm
        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut previous: HashMap<String, (String, String)> = HashMap::new(); // node -> (prev_node, rel_id)
        let mut unvisited: HashMap<String, f64> = HashMap::new();

        // Initialize
        distances.insert(from_id.to_string(), 0.0);
        unvisited.insert(from_id.to_string(), 0.0);

        while let Some(current) = unvisited
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone())
        {
            unvisited.remove(&current);

            if current == to_id {
                break;
            }

            let current_distance = distances[&current];

            // Check all neighbors
            for (_, relationship) in graph.relationships.iter() {
                let neighbor_id = match params.direction {
                    TraversalDirection::Outgoing if relationship.from_node_id == current => {
                        Some(&relationship.to_node_id)
                    }
                    TraversalDirection::Incoming if relationship.to_node_id == current => {
                        Some(&relationship.from_node_id)
                    }
                    TraversalDirection::Both => {
                        if relationship.from_node_id == current {
                            Some(&relationship.to_node_id)
                        } else if relationship.to_node_id == current {
                            Some(&relationship.from_node_id)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(neighbor) = neighbor_id {
                    // Get weight from relationship property
                    let weight = relationship
                        .properties
                        .get(weight_property)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);

                    let new_distance = current_distance + weight;

                    if new_distance < *distances.get(neighbor).unwrap_or(&f64::INFINITY) {
                        distances.insert(neighbor.clone(), new_distance);
                        previous
                            .insert(neighbor.clone(), (current.clone(), relationship.id.clone()));
                        unvisited.insert(neighbor.clone(), new_distance);
                    }
                }
            }
        }

        // Reconstruct path if found
        if distances.contains_key(to_id) {
            let mut path_nodes = Vec::new();
            let mut path_relationships = Vec::new();
            let mut edge_weights = Vec::new();

            let mut current = to_id.to_string();
            let mut path_ids = Vec::new();

            while let Some((parent_id, rel_id)) = previous.get(&current) {
                path_ids.push((parent_id.clone(), rel_id.clone(), current.clone()));
                current = parent_id.clone();
            }
            path_ids.reverse();

            // Add starting node
            if let Some(start_node) = graph.nodes.get(from_id) {
                path_nodes.push(start_node.clone());
            }

            // Add intermediate nodes, relationships, and weights
            for (_, rel_id, target_id) in path_ids {
                if let Some(rel) = graph.relationships.get(&rel_id) {
                    let weight = rel
                        .properties
                        .get(weight_property)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);

                    path_relationships.push(rel.clone());
                    edge_weights.push(weight);
                }
                if let Some(node) = graph.nodes.get(&target_id) {
                    path_nodes.push(node.clone());
                }
            }

            let path = GraphPath {
                nodes: path_nodes,
                relationships: path_relationships,
                length: edge_weights.len(),
                weight: Some(edge_weights.iter().sum()),
            };

            Ok(Some(WeightedPath::new(path, edge_weights, WeightMethod::Sum)))
        } else {
            Ok(None)
        }
    }

    async fn find_all_paths(
        &self,
        graph_id: &str,
        from_id: &str,
        to_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<GraphPath>> {
        self.ensure_graph_exists(graph_id).await?;

        // For simplicity, return just the shortest path in a vector
        // Real implementation would do DFS with path enumeration
        if let Some(path) = self
            .find_shortest_path(graph_id, from_id, to_id, params)
            .await?
        {
            Ok(vec![path])
        } else {
            Ok(vec![])
        }
    }

    async fn traverse_from(
        &self,
        graph_id: &str,
        start_id: &str,
        params: TraversalParams,
    ) -> TylResult<Vec<GraphNode>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        if !graph.nodes.contains_key(start_id) {
            return Err(graph_errors::node_not_found(start_id));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        let mut depth_map = HashMap::new();

        queue.push_back(start_id.to_string());
        visited.insert(start_id.to_string());
        depth_map.insert(start_id.to_string(), 0);

        while let Some(current_id) = queue.pop_front() {
            let current_depth = depth_map[&current_id];

            // Check depth limit
            if let Some(max_depth) = params.max_depth {
                if current_depth > max_depth {
                    continue;
                }
            }

            // Add current node to results
            if let Some(node) = graph.nodes.get(&current_id) {
                // Apply node filters
                if !params.node_labels.is_empty() {
                    let has_matching_label = node
                        .labels
                        .iter()
                        .any(|label| params.node_labels.contains(label));
                    if !has_matching_label {
                        continue;
                    }
                }

                let mut node_matches = true;
                for (key, expected_value) in &params.node_filters {
                    if let Some(actual_value) = node.properties.get(key) {
                        if actual_value != expected_value {
                            node_matches = false;
                            break;
                        }
                    } else {
                        node_matches = false;
                        break;
                    }
                }

                if node_matches {
                    result.push(node.clone());
                }

                // Continue traversal from neighbors
                if current_depth < params.max_depth.unwrap_or(usize::MAX) {
                    if let Ok(neighbors) = self
                        .get_neighbors(graph_id, &current_id, params.clone())
                        .await
                    {
                        for (neighbor_node, _) in neighbors {
                            if !visited.contains(&neighbor_node.id) {
                                visited.insert(neighbor_node.id.clone());
                                depth_map.insert(neighbor_node.id.clone(), current_depth + 1);
                                queue.push_back(neighbor_node.id);
                            }
                        }
                    }
                }
            }

            // Apply limit
            if let Some(limit) = params.limit {
                if result.len() >= limit {
                    break;
                }
            }
        }

        Ok(result)
    }

    async fn find_nodes(
        &self,
        graph_id: &str,
        labels: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<Vec<GraphNode>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut matching_nodes = Vec::new();

        for node in graph.nodes.values() {
            // Check labels
            if !labels.is_empty() {
                let has_matching_label = node.labels.iter().any(|label| labels.contains(label));
                if !has_matching_label {
                    continue;
                }
            }

            // Check properties
            let mut matches = true;
            for (key, expected_value) in &properties {
                if let Some(actual_value) = node.properties.get(key) {
                    if actual_value != expected_value {
                        matches = false;
                        break;
                    }
                } else {
                    matches = false;
                    break;
                }
            }

            if matches {
                matching_nodes.push(node.clone());
            }
        }

        Ok(matching_nodes)
    }

    async fn find_relationships(
        &self,
        graph_id: &str,
        relationship_types: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> TylResult<Vec<GraphRelationship>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut matching_relationships = Vec::new();

        for relationship in graph.relationships.values() {
            // Check relationship types
            if !relationship_types.is_empty()
                && !relationship_types.contains(&relationship.relationship_type)
            {
                continue;
            }

            // Check properties
            let mut matches = true;
            for (key, expected_value) in &properties {
                if let Some(actual_value) = relationship.properties.get(key) {
                    if actual_value != expected_value {
                        matches = false;
                        break;
                    }
                } else {
                    matches = false;
                    break;
                }
            }

            if matches {
                matching_relationships.push(relationship.clone());
            }
        }

        Ok(matching_relationships)
    }

    async fn find_nodes_temporal(
        &self,
        graph_id: &str,
        temporal_query: TemporalQuery,
    ) -> TylResult<Vec<GraphNode>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut matching_nodes = Vec::new();

        for node in graph.nodes.values() {
            if let Some(temporal_value) = node.properties.get(&temporal_query.temporal_property) {
                if let Some(timestamp_str) = temporal_value.as_str() {
                    if let Ok(timestamp) = chrono::DateTime::parse_from_rfc3339(timestamp_str) {
                        let timestamp_utc = timestamp.with_timezone(&chrono::Utc);

                        let matches = match temporal_query.operation {
                            TemporalOperation::At => {
                                temporal_query.start_time == Some(timestamp_utc)
                            }
                            TemporalOperation::Between => {
                                match (temporal_query.start_time, temporal_query.end_time) {
                                    (Some(start), Some(end)) => {
                                        timestamp_utc >= start && timestamp_utc <= end
                                    }
                                    _ => false,
                                }
                            }
                            TemporalOperation::Before => {
                                temporal_query.end_time.is_some_and(|t| timestamp_utc < t)
                            }
                            TemporalOperation::After => {
                                temporal_query.start_time.is_some_and(|t| timestamp_utc > t)
                            }
                        };

                        if matches {
                            matching_nodes.push(node.clone());
                        }
                    }
                }
            }
        }

        Ok(matching_nodes)
    }

    async fn find_relationships_temporal(
        &self,
        graph_id: &str,
        temporal_query: TemporalQuery,
    ) -> TylResult<Vec<GraphRelationship>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut matching_relationships = Vec::new();

        for relationship in graph.relationships.values() {
            if let Some(temporal_value) = relationship
                .properties
                .get(&temporal_query.temporal_property)
            {
                if let Some(timestamp_str) = temporal_value.as_str() {
                    if let Ok(timestamp) = chrono::DateTime::parse_from_rfc3339(timestamp_str) {
                        let timestamp_utc = timestamp.with_timezone(&chrono::Utc);

                        let matches = match temporal_query.operation {
                            TemporalOperation::At => {
                                temporal_query.start_time == Some(timestamp_utc)
                            }
                            TemporalOperation::Between => {
                                match (temporal_query.start_time, temporal_query.end_time) {
                                    (Some(start), Some(end)) => {
                                        timestamp_utc >= start && timestamp_utc <= end
                                    }
                                    _ => false,
                                }
                            }
                            TemporalOperation::Before => {
                                temporal_query.end_time.is_some_and(|t| timestamp_utc < t)
                            }
                            TemporalOperation::After => {
                                temporal_query.start_time.is_some_and(|t| timestamp_utc > t)
                            }
                        };

                        if matches {
                            matching_relationships.push(relationship.clone());
                        }
                    }
                }
            }
        }

        Ok(matching_relationships)
    }
}

// ===== GRAPH ANALYTICS IMPLEMENTATION =====

#[cfg(feature = "mock")]
#[async_trait]
impl GraphAnalytics for MockGraphStore {
    async fn calculate_centrality(
        &self,
        graph_id: &str,
        node_ids: Vec<String>,
        centrality_type: CentralityType,
    ) -> TylResult<HashMap<String, f64>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut centrality_scores = HashMap::new();

        let nodes_to_analyze = if node_ids.is_empty() {
            graph.nodes.keys().cloned().collect()
        } else {
            node_ids
        };

        for node_id in nodes_to_analyze {
            if !graph.nodes.contains_key(&node_id) {
                continue;
            }

            let score = match centrality_type {
                CentralityType::Degree => {
                    // Count connections
                    graph
                        .relationships
                        .values()
                        .filter(|rel| rel.from_node_id == node_id || rel.to_node_id == node_id)
                        .count() as f64
                }
                CentralityType::Betweenness => {
                    // Simplified betweenness - count shortest paths through this node
                    // Real implementation would be more complex
                    graph
                        .relationships
                        .values()
                        .filter(|rel| rel.from_node_id == node_id || rel.to_node_id == node_id)
                        .count() as f64
                        * 0.5
                }
                CentralityType::Closeness => {
                    // Simplified closeness - inverse of average distance
                    let connection_count = graph
                        .relationships
                        .values()
                        .filter(|rel| rel.from_node_id == node_id || rel.to_node_id == node_id)
                        .count();
                    if connection_count > 0 {
                        1.0 / (connection_count as f64)
                    } else {
                        0.0
                    }
                }
                CentralityType::Eigenvector => {
                    // Simplified eigenvector centrality
                    graph
                        .relationships
                        .values()
                        .filter(|rel| rel.from_node_id == node_id || rel.to_node_id == node_id)
                        .count() as f64
                        * 0.8
                }
                CentralityType::PageRank => {
                    // Simplified PageRank
                    let in_degree = graph
                        .relationships
                        .values()
                        .filter(|rel| rel.to_node_id == node_id)
                        .count();
                    0.15 + 0.85 * (in_degree as f64 / graph.nodes.len().max(1) as f64)
                }
                CentralityType::Katz => {
                    // Simplified Katz centrality
                    graph
                        .relationships
                        .values()
                        .filter(|rel| rel.from_node_id == node_id || rel.to_node_id == node_id)
                        .count() as f64
                        * 0.7
                }
            };

            centrality_scores.insert(node_id, score);
        }

        Ok(centrality_scores)
    }

    async fn detect_communities(
        &self,
        graph_id: &str,
        algorithm: ClusteringAlgorithm,
        _params: HashMap<String, serde_json::Value>,
    ) -> TylResult<HashMap<String, String>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut communities = HashMap::new();

        match algorithm {
            ClusteringAlgorithm::Louvain => {
                // Simplified community detection - group by connected components
                let mut visited = HashSet::new();
                let mut community_id = 0;

                for node_id in graph.nodes.keys() {
                    if visited.contains(node_id) {
                        continue;
                    }

                    // BFS to find connected component
                    let mut queue = VecDeque::new();
                    let community_name = format!("community_{community_id}");

                    queue.push_back(node_id.clone());
                    visited.insert(node_id.clone());
                    communities.insert(node_id.clone(), community_name.clone());

                    while let Some(current) = queue.pop_front() {
                        // Find neighbors
                        for relationship in graph.relationships.values() {
                            let neighbor = if relationship.from_node_id == current {
                                Some(&relationship.to_node_id)
                            } else if relationship.to_node_id == current {
                                Some(&relationship.from_node_id)
                            } else {
                                None
                            };

                            if let Some(neighbor_id) = neighbor {
                                if !visited.contains(neighbor_id) {
                                    visited.insert(neighbor_id.clone());
                                    communities.insert(neighbor_id.clone(), community_name.clone());
                                    queue.push_back(neighbor_id.clone());
                                }
                            }
                        }
                    }

                    community_id += 1;
                }
            }
            ClusteringAlgorithm::LabelPropagation => {
                // Simplified label propagation - assign by node labels
                for (node_id, node) in &graph.nodes {
                    let community = if let Some(primary_label) = node.labels.first() {
                        format!("label_{primary_label}")
                    } else {
                        "unlabeled".to_string()
                    };
                    communities.insert(node_id.clone(), community);
                }
            }
            ClusteringAlgorithm::ConnectedComponents => {
                // Same as Louvain for simplicity
                let mut visited = HashSet::new();
                let mut component_id = 0;

                for node_id in graph.nodes.keys() {
                    if visited.contains(node_id) {
                        continue;
                    }

                    let mut queue = VecDeque::new();
                    let component_name = format!("component_{component_id}");

                    queue.push_back(node_id.clone());
                    visited.insert(node_id.clone());
                    communities.insert(node_id.clone(), component_name.clone());

                    while let Some(current) = queue.pop_front() {
                        for relationship in graph.relationships.values() {
                            let neighbor = if relationship.from_node_id == current {
                                Some(&relationship.to_node_id)
                            } else if relationship.to_node_id == current {
                                Some(&relationship.from_node_id)
                            } else {
                                None
                            };

                            if let Some(neighbor_id) = neighbor {
                                if !visited.contains(neighbor_id) {
                                    visited.insert(neighbor_id.clone());
                                    communities.insert(neighbor_id.clone(), component_name.clone());
                                    queue.push_back(neighbor_id.clone());
                                }
                            }
                        }
                    }

                    component_id += 1;
                }
            }
            ClusteringAlgorithm::Leiden => {
                // Simplified Leiden algorithm - similar to Louvain but with refinement
                let mut visited = HashSet::new();
                let mut community_id = 0;
                for node_id in graph.nodes.keys() {
                    if visited.contains(node_id) {
                        continue;
                    }
                    // BFS to find connected component with refinement
                    let mut queue = VecDeque::new();
                    let community_name = format!("leiden_community_{community_id}");

                    queue.push_back(node_id.clone());
                    visited.insert(node_id.clone());
                    communities.insert(node_id.clone(), community_name.clone());
                    while let Some(current) = queue.pop_front() {
                        // Find neighbors with higher connection strength
                        for relationship in graph.relationships.values() {
                            let neighbor = if relationship.from_node_id == current {
                                Some(&relationship.to_node_id)
                            } else if relationship.to_node_id == current {
                                Some(&relationship.from_node_id)
                            } else {
                                None
                            };
                            if let Some(neighbor_id) = neighbor {
                                if !visited.contains(neighbor_id) {
                                    visited.insert(neighbor_id.clone());
                                    communities.insert(neighbor_id.clone(), community_name.clone());
                                    queue.push_back(neighbor_id.clone());
                                }
                            }
                        }
                    }
                    community_id += 1;
                }
            }
        }

        Ok(communities)
    }

    async fn find_patterns(
        &self,
        graph_id: &str,
        pattern_size: usize,
        min_frequency: usize,
    ) -> TylResult<Vec<(GraphPath, usize)>> {
        self.ensure_graph_exists(graph_id).await?;

        if pattern_size < 2 {
            return Ok(vec![]);
        }

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        // Simplified pattern detection - find common relationship chains
        let mut pattern_counts: HashMap<Vec<String>, usize> = HashMap::new();

        // Find all paths of the specified length
        for start_node_id in graph.nodes.keys() {
            let params = TraversalParams::new().with_max_depth(pattern_size - 1);

            if let Ok(neighbors) = self.get_neighbors(graph_id, start_node_id, params).await {
                for (_, relationship) in neighbors {
                    // Create a simple pattern signature
                    let pattern = vec![relationship.relationship_type.clone()];
                    *pattern_counts.entry(pattern).or_insert(0) += 1;
                }
            }
        }

        // Convert to results with frequency filter
        let mut patterns = Vec::new();
        for (pattern_sig, count) in pattern_counts {
            if count >= min_frequency {
                // Create a dummy path for the pattern
                let dummy_node1 = GraphNode::new().with_label("Pattern");
                let dummy_node2 = GraphNode::new().with_label("Pattern");
                let dummy_rel = GraphRelationship::new("p1", "p2", &pattern_sig[0]);

                let path = GraphPath {
                    nodes: vec![dummy_node1, dummy_node2],
                    relationships: vec![dummy_rel],
                    length: 1,
                    weight: Some(1.0),
                };

                patterns.push((path, count));
            }
        }

        Ok(patterns)
    }

    async fn recommend_relationships(
        &self,
        graph_id: &str,
        node_id: &str,
        recommendation_type: RecommendationType,
        limit: usize,
    ) -> TylResult<Vec<(String, String, f64)>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        if !graph.nodes.contains_key(node_id) {
            return Err(graph_errors::node_not_found(node_id));
        }

        let mut recommendations = Vec::new();

        match recommendation_type {
            RecommendationType::CommonNeighbors => {
                // Find nodes that share neighbors with the target node
                let mut neighbor_counts: HashMap<String, usize> = HashMap::new();

                // Get target node's neighbors
                let target_neighbors: HashSet<String> = graph
                    .relationships
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

                // For each neighbor, find their neighbors
                for neighbor_id in &target_neighbors {
                    let second_neighbors: HashSet<String> = graph
                        .relationships
                        .values()
                        .filter_map(|rel| {
                            if rel.from_node_id == *neighbor_id {
                                Some(rel.to_node_id.clone())
                            } else if rel.to_node_id == *neighbor_id {
                                Some(rel.from_node_id.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    for second_neighbor in second_neighbors {
                        if second_neighbor != node_id
                            && !target_neighbors.contains(&second_neighbor)
                        {
                            *neighbor_counts.entry(second_neighbor).or_insert(0) += 1;
                        }
                    }
                }

                // Convert to recommendations
                for (candidate_id, common_count) in neighbor_counts {
                    let confidence = common_count as f64 / target_neighbors.len().max(1) as f64;
                    recommendations.push((candidate_id, "RECOMMENDED".to_string(), confidence));
                }
            }
            RecommendationType::SimilarNodes => {
                // Find nodes with similar properties or labels
                if let Some(target_node) = graph.nodes.get(node_id) {
                    for (candidate_id, candidate_node) in &graph.nodes {
                        if candidate_id == node_id {
                            continue;
                        }

                        // Check if already connected
                        let already_connected = graph.relationships.values().any(|rel| {
                            (rel.from_node_id == node_id && rel.to_node_id == *candidate_id)
                                || (rel.to_node_id == node_id && rel.from_node_id == *candidate_id)
                        });

                        if already_connected {
                            continue;
                        }

                        // Calculate similarity based on shared labels
                        let shared_labels: HashSet<_> = target_node
                            .labels
                            .iter()
                            .filter(|label| candidate_node.labels.contains(label))
                            .collect();

                        let total_labels = target_node.labels.len() + candidate_node.labels.len();
                        if total_labels > 0 {
                            let similarity = (shared_labels.len() * 2) as f64 / total_labels as f64;
                            if similarity > 0.0 {
                                recommendations.push((
                                    candidate_id.clone(),
                                    "SIMILAR".to_string(),
                                    similarity,
                                ));
                            }
                        }
                    }
                }
            }
            RecommendationType::StructuralEquivalence => {
                // Simplified structural equivalence - nodes with similar degree
                let target_degree = graph
                    .relationships
                    .values()
                    .filter(|rel| rel.from_node_id == node_id || rel.to_node_id == node_id)
                    .count();

                for candidate_id in graph.nodes.keys() {
                    if candidate_id == node_id {
                        continue;
                    }

                    let candidate_degree = graph
                        .relationships
                        .values()
                        .filter(|rel| {
                            rel.from_node_id == *candidate_id || rel.to_node_id == *candidate_id
                        })
                        .count();

                    let degree_similarity = if target_degree.max(candidate_degree) > 0 {
                        1.0 - ((target_degree as i32 - candidate_degree as i32).abs() as f64
                            / target_degree.max(candidate_degree) as f64)
                    } else {
                        1.0
                    };

                    if degree_similarity > 0.5 {
                        recommendations.push((
                            candidate_id.clone(),
                            "STRUCTURALLY_SIMILAR".to_string(),
                            degree_similarity,
                        ));
                    }
                }
            }
            RecommendationType::PathSimilarity => {
                // Find nodes at similar distances from common nodes
                let mut path_similarities: HashMap<String, f64> = HashMap::new();

                for candidate_id in graph.nodes.keys() {
                    if candidate_id == node_id {
                        continue;
                    }

                    // Check if there's a path between them
                    let params = TraversalParams::default();
                    if let Ok(Some(_)) = self
                        .find_shortest_path(graph_id, node_id, candidate_id, params)
                        .await
                    {
                        path_similarities.insert(candidate_id.clone(), 0.8);
                    }
                }

                for (candidate_id, similarity) in path_similarities {
                    recommendations.push((candidate_id, "PATH_SIMILAR".to_string(), similarity));
                }
            }
        }

        // Sort by confidence and limit
        recommendations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        recommendations.truncate(limit);

        Ok(recommendations)
    }

    async fn execute_aggregation(
        &self,
        graph_id: &str,
        aggregation_query: AggregationQuery,
    ) -> TylResult<Vec<AggregationResult>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut results = Vec::new();

        // Simple aggregation implementation
        match aggregation_query.function {
            AggregationFunction::Count => {
                if aggregation_query.group_by.is_empty() {
                    // Simple count
                    let count = graph.nodes.len();
                    results.push(AggregationResult {
                        value: serde_json::json!(count),
                        groups: HashMap::new(),
                        metadata: HashMap::new(),
                    });
                } else {
                    // Group by and count
                    let mut groups: HashMap<String, usize> = HashMap::new();

                    for node in graph.nodes.values() {
                        for group_property in &aggregation_query.group_by {
                            if let Some(group_value) = node.properties.get(group_property) {
                                let group_key = group_value.to_string();
                                *groups.entry(group_key).or_insert(0) += 1;
                            }
                        }
                    }

                    for (group_key, count) in groups {
                        let mut group_map = HashMap::new();
                        group_map.insert(
                            aggregation_query.group_by[0].clone(),
                            serde_json::json!(group_key),
                        );

                        results.push(AggregationResult {
                            value: serde_json::json!(count),
                            groups: group_map,
                            metadata: HashMap::new(),
                        });
                    }
                }
            }
            AggregationFunction::Sum | AggregationFunction::Avg => {
                if let Some(property) = &aggregation_query.property {
                    let mut values = Vec::new();

                    for node in graph.nodes.values() {
                        if let Some(prop_value) = node.properties.get(property) {
                            if let Some(num_value) = prop_value.as_f64() {
                                values.push(num_value);
                            }
                        }
                    }

                    let result_value = match aggregation_query.function {
                        AggregationFunction::Sum => values.iter().sum::<f64>(),
                        AggregationFunction::Avg => {
                            if values.is_empty() {
                                0.0
                            } else {
                                values.iter().sum::<f64>() / values.len() as f64
                            }
                        }
                        _ => unreachable!(),
                    };

                    results.push(AggregationResult {
                        value: serde_json::json!(result_value),
                        groups: HashMap::new(),
                        metadata: HashMap::new(),
                    });
                }
            }
            AggregationFunction::Min | AggregationFunction::Max => {
                if let Some(property) = &aggregation_query.property {
                    let mut values = Vec::new();

                    for node in graph.nodes.values() {
                        if let Some(prop_value) = node.properties.get(property) {
                            if let Some(num_value) = prop_value.as_f64() {
                                values.push(num_value);
                            }
                        }
                    }

                    if let Some(result_value) = match aggregation_query.function {
                        AggregationFunction::Min => values
                            .iter()
                            .fold(None, |acc, &x| Some(acc.map_or(x, |y: f64| y.min(x)))),
                        AggregationFunction::Max => values
                            .iter()
                            .fold(None, |acc, &x| Some(acc.map_or(x, |y: f64| y.max(x)))),
                        _ => unreachable!(),
                    } {
                        results.push(AggregationResult {
                            value: serde_json::json!(result_value),
                            groups: HashMap::new(),
                            metadata: HashMap::new(),
                        });
                    }
                }
            }
            AggregationFunction::Collect => {
                if let Some(property) = &aggregation_query.property {
                    let mut values = Vec::new();

                    for node in graph.nodes.values() {
                        if let Some(prop_value) = node.properties.get(property) {
                            values.push(prop_value.clone());
                        }
                    }

                    results.push(AggregationResult {
                        value: serde_json::json!(values),
                        groups: HashMap::new(),
                        metadata: HashMap::new(),
                    });
                }
            }
            AggregationFunction::StdDev => {
                if let Some(property) = &aggregation_query.property {
                    let mut values = Vec::new();

                    for node in graph.nodes.values() {
                        if let Some(prop_value) = node.properties.get(property) {
                            if let Some(num_value) = prop_value.as_f64() {
                                values.push(num_value);
                            }
                        }
                    }

                    if !values.is_empty() {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / values.len() as f64;
                        let std_dev = variance.sqrt();

                        results.push(AggregationResult {
                            value: serde_json::json!(std_dev),
                            groups: HashMap::new(),
                            metadata: HashMap::new(),
                        });
                    }
                }
            }
        }

        Ok(results)
    }
}

// ===== REMAINING TRAIT IMPLEMENTATIONS =====

#[cfg(feature = "mock")]
#[async_trait]
impl GraphQueryExecutor for MockGraphStore {
    async fn execute_query(&self, graph_id: &str, query: GraphQuery) -> TylResult<QueryResult> {
        self.ensure_graph_exists(graph_id).await?;

        // Mock implementation - just return the query back as data
        let mut metadata = HashMap::new();
        metadata.insert("executed_at".to_string(), serde_json::json!(chrono::Utc::now()));
        metadata.insert("graph_id".to_string(), serde_json::json!(graph_id));
        metadata.insert(
            "query_type".to_string(),
            serde_json::json!(if query.is_write_query {
                "write"
            } else {
                "read"
            }),
        );

        Ok(QueryResult {
            data: vec![{
                let mut data_row = HashMap::new();
                data_row.insert("query".to_string(), serde_json::json!(query.query));
                data_row.insert("status".to_string(), serde_json::json!("executed"));
                data_row
            }],
            metadata,
        })
    }

    async fn execute_read_query(
        &self,
        graph_id: &str,
        query: GraphQuery,
    ) -> TylResult<QueryResult> {
        if query.is_write_query {
            return Err(TylError::validation("query_type", "Expected read query"));
        }

        self.execute_query(graph_id, query).await
    }

    async fn execute_write_query(
        &self,
        graph_id: &str,
        query: GraphQuery,
    ) -> TylResult<QueryResult> {
        if !query.is_write_query {
            return Err(TylError::validation("query_type", "Expected write query"));
        }

        self.execute_query(graph_id, query).await
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphTransaction for MockGraphStore {
    async fn begin_transaction(
        &self,
        graph_id: &str,
        mut context: TransactionContext,
    ) -> TylResult<TransactionContext> {
        self.ensure_graph_exists(graph_id).await?;

        // Store graph_id in metadata
        context
            .metadata
            .insert("graph_id".to_string(), serde_json::json!(graph_id));

        let mut transactions = self.transactions.write().await;
        transactions.insert(context.id.clone(), context.clone());

        Ok(context)
    }

    async fn commit_transaction(&self, graph_id: &str, transaction_id: &str) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut transactions = self.transactions.write().await;
        transactions.remove(transaction_id);

        Ok(())
    }

    async fn rollback_transaction(&self, graph_id: &str, transaction_id: &str) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut transactions = self.transactions.write().await;
        transactions.remove(transaction_id);

        Ok(())
    }

    async fn get_transaction_status(
        &self,
        graph_id: &str,
        transaction_id: &str,
    ) -> TylResult<Option<TransactionContext>> {
        self.ensure_graph_exists(graph_id).await?;

        let transactions = self.transactions.read().await;
        Ok(transactions.get(transaction_id).cloned())
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphIndexManager for MockGraphStore {
    async fn create_index(&self, graph_id: &str, index_config: IndexConfig) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut indexes = self.indexes.write().await;
        let graph_indexes = indexes.entry(graph_id.to_string()).or_insert_with(Vec::new);

        // Check if index with same name already exists
        if graph_indexes
            .iter()
            .any(|idx| idx.name == index_config.name)
        {
            return Err(TylError::validation(
                "index_name",
                format!("Index '{}' already exists", index_config.name),
            ));
        }

        graph_indexes.push(index_config);
        Ok(())
    }

    async fn drop_index(&self, graph_id: &str, index_name: &str) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut indexes = self.indexes.write().await;
        if let Some(graph_indexes) = indexes.get_mut(graph_id) {
            graph_indexes.retain(|idx| idx.name != index_name);
        }

        Ok(())
    }

    async fn list_indexes(&self, graph_id: &str) -> TylResult<Vec<IndexConfig>> {
        self.ensure_graph_exists(graph_id).await?;

        let indexes = self.indexes.read().await;
        Ok(indexes.get(graph_id).cloned().unwrap_or_default())
    }

    async fn get_index(&self, graph_id: &str, index_name: &str) -> TylResult<Option<IndexConfig>> {
        self.ensure_graph_exists(graph_id).await?;

        let indexes = self.indexes.read().await;
        if let Some(graph_indexes) = indexes.get(graph_id) {
            Ok(graph_indexes
                .iter()
                .find(|idx| idx.name == index_name)
                .cloned())
        } else {
            Ok(None)
        }
    }

    async fn rebuild_index(&self, graph_id: &str, index_name: &str) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        // For mock, we just verify the index exists
        let indexes = self.indexes.read().await;
        if let Some(graph_indexes) = indexes.get(graph_id) {
            if graph_indexes.iter().any(|idx| idx.name == index_name) {
                Ok(())
            } else {
                Err(TylError::not_found("index", index_name))
            }
        } else {
            Err(TylError::not_found("index", index_name))
        }
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphConstraintManager for MockGraphStore {
    async fn create_constraint(
        &self,
        graph_id: &str,
        constraint_config: ConstraintConfig,
    ) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut constraints = self.constraints.write().await;
        let graph_constraints = constraints
            .entry(graph_id.to_string())
            .or_insert_with(Vec::new);

        // Check if constraint with same name already exists
        if graph_constraints
            .iter()
            .any(|c| c.name == constraint_config.name)
        {
            return Err(TylError::validation(
                "constraint_name",
                format!("Constraint '{}' already exists", constraint_config.name),
            ));
        }

        // For uniqueness constraints, validate existing data
        if constraint_config.constraint_type == ConstraintType::Unique {
            let graphs = self.graphs.read().await;
            if let Some(graph) = graphs.get(graph_id) {
                for property in &constraint_config.properties {
                    let mut seen_values = HashSet::new();

                    for node in graph.nodes.values() {
                        if constraint_config.labels_or_types.is_empty()
                            || node
                                .labels
                                .iter()
                                .any(|l| constraint_config.labels_or_types.contains(l))
                        {
                            if let Some(value) = node.properties.get(property) {
                                if !seen_values.insert(value.clone()) {
                                    return Err(TylError::validation(
                                        "unique_constraint",
                                        format!("Duplicate value found for property '{property}"),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        graph_constraints.push(constraint_config);
        Ok(())
    }

    async fn drop_constraint(&self, graph_id: &str, constraint_name: &str) -> TylResult<()> {
        self.ensure_graph_exists(graph_id).await?;

        let mut constraints = self.constraints.write().await;
        if let Some(graph_constraints) = constraints.get_mut(graph_id) {
            graph_constraints.retain(|c| c.name != constraint_name);
        }

        Ok(())
    }

    async fn list_constraints(&self, graph_id: &str) -> TylResult<Vec<ConstraintConfig>> {
        self.ensure_graph_exists(graph_id).await?;

        let constraints = self.constraints.read().await;
        Ok(constraints.get(graph_id).cloned().unwrap_or_default())
    }

    async fn get_constraint(
        &self,
        graph_id: &str,
        constraint_name: &str,
    ) -> TylResult<Option<ConstraintConfig>> {
        self.ensure_graph_exists(graph_id).await?;

        let constraints = self.constraints.read().await;
        if let Some(graph_constraints) = constraints.get(graph_id) {
            Ok(graph_constraints
                .iter()
                .find(|c| c.name == constraint_name)
                .cloned())
        } else {
            Ok(None)
        }
    }

    async fn validate_constraints(
        &self,
        graph_id: &str,
    ) -> TylResult<Vec<HashMap<String, serde_json::Value>>> {
        self.ensure_graph_exists(graph_id).await?;

        let constraints = self.constraints.read().await;
        let graphs = self.graphs.read().await;

        let mut violations = Vec::new();

        if let (Some(graph_constraints), Some(graph)) =
            (constraints.get(graph_id), graphs.get(graph_id))
        {
            for constraint in graph_constraints {
                match constraint.constraint_type {
                    ConstraintType::Unique => {
                        for property in &constraint.properties {
                            let mut value_to_nodes: HashMap<serde_json::Value, Vec<String>> =
                                HashMap::new();

                            for (node_id, node) in &graph.nodes {
                                if constraint.labels_or_types.is_empty()
                                    || node
                                        .labels
                                        .iter()
                                        .any(|l| constraint.labels_or_types.contains(l))
                                {
                                    if let Some(value) = node.properties.get(property) {
                                        value_to_nodes
                                            .entry(value.clone())
                                            .or_default()
                                            .push(node_id.clone());
                                    }
                                }
                            }

                            for (value, node_ids) in value_to_nodes {
                                if node_ids.len() > 1 {
                                    let mut violation = HashMap::new();
                                    violation.insert(
                                        "constraint".to_string(),
                                        serde_json::json!(constraint.name),
                                    );
                                    violation
                                        .insert("type".to_string(), serde_json::json!("unique"));
                                    violation.insert(
                                        "property".to_string(),
                                        serde_json::json!(property),
                                    );
                                    violation.insert("value".to_string(), value);
                                    violation
                                        .insert("nodes".to_string(), serde_json::json!(node_ids));
                                    violations.push(violation);
                                }
                            }
                        }
                    }
                    ConstraintType::Exists => {
                        for property in &constraint.properties {
                            for (node_id, node) in &graph.nodes {
                                if (constraint.labels_or_types.is_empty()
                                    || node
                                        .labels
                                        .iter()
                                        .any(|l| constraint.labels_or_types.contains(l)))
                                    && !node.properties.contains_key(property)
                                {
                                    let mut violation = HashMap::new();
                                    violation.insert(
                                        "constraint".to_string(),
                                        serde_json::json!(constraint.name),
                                    );
                                    violation.insert(
                                        "type".to_string(),
                                        serde_json::json!("exists"),
                                    );
                                    violation.insert(
                                        "property".to_string(),
                                        serde_json::json!(property),
                                    );
                                    violation
                                        .insert("node".to_string(), serde_json::json!(node_id));
                                    violations.push(violation);
                                }
                            }
                        }
                    }
                    _ => {
                        // Other constraint types not implemented in mock
                    }
                }
            }
        }

        Ok(violations)
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphBulkOperations for MockGraphStore {
    async fn bulk_create_nodes(
        &self,
        graph_id: &str,
        bulk_operation: BulkOperation<GraphNode>,
    ) -> TylResult<Vec<Result<String, TylError>>> {
        // Use existing batch implementation
        self.create_nodes_batch(graph_id, bulk_operation.items)
            .await
    }

    async fn bulk_create_relationships(
        &self,
        graph_id: &str,
        bulk_operation: BulkOperation<GraphRelationship>,
    ) -> TylResult<Vec<Result<String, TylError>>> {
        let mut results = Vec::new();
        for relationship in bulk_operation.items {
            match self.create_relationship(graph_id, relationship).await {
                Ok(id) => results.push(Ok(id)),
                Err(e) => {
                    if bulk_operation.continue_on_error {
                        results.push(Err(e));
                    } else {
                        return Ok(results);
                    }
                }
            }
        }
        Ok(results)
    }

    async fn bulk_update_nodes(
        &self,
        graph_id: &str,
        updates: HashMap<String, HashMap<String, serde_json::Value>>,
    ) -> TylResult<Vec<Result<(), TylError>>> {
        let mut results = Vec::new();
        for (node_id, properties) in updates {
            match self.update_node(graph_id, &node_id, properties).await {
                Ok(_) => results.push(Ok(())),
                Err(e) => results.push(Err(e)),
            }
        }
        Ok(results)
    }

    async fn bulk_delete_nodes(
        &self,
        graph_id: &str,
        node_ids: Vec<String>,
    ) -> TylResult<Vec<Result<(), TylError>>> {
        let mut results = Vec::new();
        for node_id in node_ids {
            match self.delete_node(graph_id, &node_id).await {
                Ok(_) => results.push(Ok(())),
                Err(e) => results.push(Err(e)),
            }
        }
        Ok(results)
    }

    async fn import_data(
        &self,
        graph_id: &str,
        import_format: &str,
        data: Vec<u8>,
    ) -> TylResult<HashMap<String, serde_json::Value>> {
        self.ensure_graph_exists(graph_id).await?;

        let mut result = HashMap::new();
        result.insert("format".to_string(), serde_json::json!(import_format));
        result.insert("size_bytes".to_string(), serde_json::json!(data.len()));
        result.insert("status".to_string(), serde_json::json!("success"));
        result.insert("nodes_imported".to_string(), serde_json::json!(0));
        result.insert("relationships_imported".to_string(), serde_json::json!(0));

        // Mock implementation - in real version would parse data based on format
        Ok(result)
    }

    async fn export_data(
        &self,
        graph_id: &str,
        export_format: &str,
        _export_params: HashMap<String, serde_json::Value>,
    ) -> TylResult<Vec<u8>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        match export_format.to_lowercase().as_str() {
            "json" => {
                let export_data = serde_json::json!({
                    "nodes": graph.nodes.values().collect::<Vec<_>>(),
                    "relationships": graph.relationships.values().collect::<Vec<_>>(),
                    "metadata": {
                        "graph_id": graph_id,
                        "node_count": graph.nodes.len(),
                        "relationship_count": graph.relationships.len(),
                        "exported_at": chrono::Utc::now()
                    }
                });

                Ok(serde_json::to_vec(&export_data).unwrap())
            }
            "csv" => {
                // Simple CSV export - just node IDs and labels
                let mut csv_data = String::new();
                csv_data.push_str("id,labels,properties\n");

                for node in graph.nodes.values() {
                    csv_data.push_str(&format!(
                        "{},{},{}\n",
                        node.id,
                        node.labels.join(";"),
                        serde_json::to_string(&node.properties).unwrap_or_default()
                    ));
                }

                Ok(csv_data.into_bytes())
            }
            _ => Err(TylError::validation(
                "export_format",
                format!("Unsupported format: {export_format}"),
            )),
        }
    }
}

#[cfg(feature = "mock")]
#[async_trait]
impl GraphHealth for MockGraphStore {
    async fn is_healthy(&self) -> TylResult<bool> {
        Ok(true)
    }

    async fn health_check(&self) -> TylResult<HashMap<String, serde_json::Value>> {
        let mut health_info = HashMap::new();

        health_info.insert("status".to_string(), serde_json::json!("healthy"));
        health_info.insert("type".to_string(), serde_json::json!("mock"));
        health_info.insert("version".to_string(), serde_json::json!("1.0.0"));
        health_info.insert("uptime_seconds".to_string(), serde_json::json!(0));
        health_info
            .insert("total_graphs".to_string(), serde_json::json!(self.graphs.read().await.len()));
        health_info
            .insert("total_nodes".to_string(), serde_json::json!(self.total_node_count().await));
        health_info.insert(
            "total_relationships".to_string(),
            serde_json::json!(self.total_relationship_count().await),
        );

        Ok(health_info)
    }

    async fn get_graph_statistics(
        &self,
        graph_id: &str,
    ) -> TylResult<HashMap<String, serde_json::Value>> {
        self.ensure_graph_exists(graph_id).await?;

        let graphs = self.graphs.read().await;
        let graph = graphs.get(graph_id).unwrap();

        let mut stats = HashMap::new();
        stats.insert("node_count".to_string(), serde_json::json!(graph.nodes.len()));
        stats
            .insert("relationship_count".to_string(), serde_json::json!(graph.relationships.len()));

        // Collect label statistics
        let mut labels: HashMap<String, usize> = HashMap::new();
        for node in graph.nodes.values() {
            for label in &node.labels {
                *labels.entry(label.clone()).or_insert(0) += 1;
            }
        }
        stats.insert("labels".to_string(), serde_json::to_value(labels).unwrap());

        // Collect relationship type statistics
        let mut rel_types: HashMap<String, usize> = HashMap::new();
        for relationship in graph.relationships.values() {
            *rel_types
                .entry(relationship.relationship_type.clone())
                .or_insert(0) += 1;
        }
        stats.insert("relationship_types".to_string(), serde_json::to_value(rel_types).unwrap());

        Ok(stats)
    }

    async fn get_all_statistics(
        &self,
    ) -> TylResult<HashMap<String, HashMap<String, serde_json::Value>>> {
        let graphs = self.graphs.read().await;
        let mut all_stats = HashMap::new();

        for graph_id in graphs.keys() {
            if let Ok(stats) = self.get_graph_statistics(graph_id).await {
                all_stats.insert(graph_id.clone(), stats);
            }
        }

        Ok(all_stats)
    }
}
