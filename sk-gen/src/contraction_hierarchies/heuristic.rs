use std::collections::HashMap;

use petgraph::graph::{
    Graph,
    NodeIndex,
};

/// A trait providing contraction hierarchy functionality for graphs.
pub trait NestedDissection<N, E> {
    /// Generates a contraction order for the graph nodes using nested dissection.
    ///
    /// This helps optimize the contraction process by minimizing the number of shortcuts needed.
    fn contraction_order(&self) -> Vec<NodeIndex>;

    /// Recursively generates a partition tree for the given nodes.
    ///
    /// The partition tree represents the hierarchical decomposition of the graph, which is used
    /// to push the separator set to the end of the contraction order at each recursive level.
    fn partition_tree(&self, nodes: Vec<NodeIndex>) -> PartitionTreeNode;
}

impl<N, E> NestedDissection<N, E> for Graph<N, E> {
    fn contraction_order(&self) -> Vec<NodeIndex> {
        self.partition_tree(self.node_indices().collect()).to_vec()
    }

    fn partition_tree(&self, nodes: Vec<NodeIndex>) -> PartitionTreeNode {
        if nodes.len() <= 1 {
            return PartitionTreeNode { a: None, b: None, separator: nodes };
        }

        let mut node_to_metis = HashMap::new();
        let mut metis_to_node = Vec::with_capacity(nodes.len());
        for (i, &node) in nodes.iter().enumerate() {
            node_to_metis.insert(node, i);
            metis_to_node.push(node);
        }

        // METIS input: CSR for the `nodes` induced subgraph.
        let mut xadj = Vec::with_capacity(nodes.len() + 1);
        let mut adjncy = Vec::new();
        let mut current_idx = 0;
        xadj.push(current_idx);
        for &node in &nodes {
            for neighbor in self.neighbors(node) {
                if nodes.contains(&neighbor) {
                    let metis_idx = node_to_metis[&neighbor];
                    adjncy.push(metis::Idx::try_from(metis_idx).unwrap());
                    current_idx += 1;
                }
            }
            xadj.push(current_idx);
        }

        // Partition the graph into a, b
        let mut part = vec![0; nodes.len()];
        let metis_graph = metis::Graph::new(1, 2, &xadj, &adjncy).expect("Failed to create METIS graph");
        metis_graph.part_recursive(&mut part).expect("Failed to partition graph");

        let mut a = Vec::new();
        let mut b = Vec::new();

        // partition into a, b, and separator
        for (metis_idx, &p) in part.iter().enumerate() {
            let node = metis_to_node[metis_idx];
            match p {
                0 => a.push(node),
                1 => b.push(node),
                _ => panic!("METIS returned an invalid partition"),
            }
        }

        // TODO: separator should not be empty (by construction)
        if a.is_empty() || b.is_empty() {
            // if the partition fails to create two non-empty sets, default to empty separator
            let (left, right) = nodes.split_at(nodes.len() / 2);
            a = left.to_vec();
            b = right.to_vec();
        }

        let a_tree = if a.is_empty() { None } else { Some(Box::new(self.partition_tree(a))) };

        let b_tree = if b.is_empty() { None } else { Some(Box::new(self.partition_tree(b))) };

        let separator = Vec::new();

        PartitionTreeNode { a: a_tree, b: b_tree, separator }
    }
}

/// A tree structure representing the hierarchical partitioning of a graph.
///
/// Used as part of the nested dissection algorithm to generate contraction orders.
#[derive(Debug)]
pub struct PartitionTreeNode {
    /// The first partition of nodes.
    pub a: Option<Box<PartitionTreeNode>>,
    /// The second partition of nodes.
    pub b: Option<Box<PartitionTreeNode>>,
    /// The separator nodes.
    pub separator: Vec<NodeIndex>,
}

impl PartitionTreeNode {
    /// Flattens the recursive ordering into a single vector.
    pub fn to_vec(&self) -> Vec<NodeIndex> {
        let mut result = Vec::new();

        if let Some(a) = &self.a {
            result.extend(a.to_vec());
        }

        if let Some(b) = &self.b {
            result.extend(b.to_vec());
        }

        result.extend(&self.separator);

        result
    }
}
