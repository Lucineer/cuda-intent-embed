//! Intent Embedding System
//! Transforms human text into structured intent vectors for A2A deliberation.

/// Intent vector: the structured representation of what a human wants
#[derive(Debug, Clone)]
pub struct IntentVector {
    /// Primary goal extracted from text
    pub goal: String,
    /// Hard constraints (must be satisfied)
    pub hard_constraints: Vec<Constraint>,
    /// Soft constraints (nice to have)
    pub soft_constraints: Vec<Constraint>,
    /// Domain classification
    pub domain: Domain,
    /// Complexity estimate (0.0-1.0)
    pub complexity: f64,
    /// Embedding dimensions for the intent
    pub dimensions: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub text: String,
    pub kind: ConstraintKind,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    Ordering,
    DataType,
    Performance,
    Safety,
    Compatibility,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Domain {
    Sorting,
    Filtering,
    Transforming,
    Aggregating,
    Searching,
    Validating,
    Creating,
    Unknown,
}

/// Intent parser: extracts structured intent from human text
pub struct IntentParser {
    domain_keywords: Vec<(Domain, Vec<&'static str>)>,
    constraint_keywords: Vec<(ConstraintKind, Vec<&'static str>)>,
}

impl IntentParser {
    pub fn new() -> Self {
        Self {
            domain_keywords: vec![
                (Domain::Sorting, vec!["sort", "order", "rank", "arrange"]),
                (Domain::Filtering, vec!["filter", "only", "exclude", "where"]),
                (Domain::Transforming, vec!["convert", "transform", "map", "change"]),
                (Domain::Aggregating, vec!["count", "sum", "average", "total"]),
                (Domain::Searching, vec!["find", "search", "locate", "lookup"]),
                (Domain::Validating, vec!["check", "verify", "validate"]),
                (Domain::Creating, vec!["create", "build", "make", "generate"]),
            ],
            constraint_keywords: vec![
                (ConstraintKind::Ordering, vec!["ascending", "descending", "reverse"]),
                (ConstraintKind::DataType, vec!["numbers", "strings", "integers", "list", "array"]),
                (ConstraintKind::Performance, vec!["fast", "efficient", "optimize", "o(n)"]),
                (ConstraintKind::Safety, vec!["safe", "handle error", "validate input"]),
                (ConstraintKind::Compatibility, vec!["utf8", "unicode", "cross-platform"]),
            ],
        }
    }

    /// Parse human text into a structured intent vector
    pub fn parse(&self, text: &str) -> IntentVector {
        let lower = text.to_lowercase();
        let goal = text.trim().to_string();

        let domain = self.classify_domain(&lower);
        let (hard, soft) = self.extract_constraints(&lower);
        let complexity = self.estimate_complexity(&goal);
        let dimensions = self.compute_dimensions(&domain, &hard, &soft, complexity);

        IntentVector { goal, hard_constraints: hard, soft_constraints: soft,
                       domain, complexity, dimensions }
    }

    fn classify_domain(&self, text: &str) -> Domain {
        let mut best = (Domain::Unknown, 0usize);
        for (domain, keywords) in &self.domain_keywords {
            let matches = keywords.iter().filter(|kw| text.contains(**kw)).count();
            if matches > best.1 { best = (domain.clone(), matches); }
        }
        best.0
    }

    fn extract_constraints(&self, text: &str) -> (Vec<Constraint>, Vec<Constraint>) {
        let mut hard = vec![];
        let mut soft = vec![];
        for (kind, keywords) in &self.constraint_keywords {
            for kw in keywords {
                if text.contains(*kw) {
                    let c = Constraint {
                        text: kw.to_string(), kind: kind.clone(),
                        weight: if kind == &ConstraintKind::Ordering { 1.0 } else { 0.7 },
                    };
                    if kind == &ConstraintKind::Ordering || kind == &ConstraintKind::DataType {
                        hard.push(c);
                    } else {
                        soft.push(c);
                    }
                }
            }
        }
        (hard, soft)
    }

    fn estimate_complexity(&self, text: &str) -> f64 {
        let words = text.split_whitespace().count();
        let clauses = text.matches(&[',', 'and', 'or', 'but'][..]).count();
        let base = 0.1;
        let word_factor = (words as f64 * 0.02).min(0.3);
        let clause_factor = (clauses as f64 * 0.05).min(0.2);
        let has_multi = text.contains(" and then ") || text.contains(" after ");
        let multi_factor = if has_multi { 0.2 } else { 0.0 };
        (base + word_factor + clause_factor + multi_factor).min(1.0)
    }

    fn compute_dimensions(&self, domain: &Domain, hard: &[Constraint], soft: &[Constraint], complexity: f64) -> Vec<f64> {
        // 8-dimensional intent space
        let domain_idx = match domain {
            Domain::Sorting => 0.0, Domain::Filtering => 1.0, Domain::Transforming => 2.0,
            Domain::Aggregating => 3.0, Domain::Searching => 4.0, Domain::Validating => 5.0,
            Domain::Creating => 6.0, Domain::Unknown => 7.0,
        };
        let constraint_count = (hard.len() + soft.len()) as f64;
        vec![
            domain_idx / 7.0,           // dim 0: domain (normalized)
            constraint_count / 10.0,    // dim 1: constraint density
            hard.len() as f64 / 5.0,    // dim 2: hard constraint ratio
            complexity,                 // dim 3: complexity
            if hard.is_empty() { 0.0 } else { hard.iter().map(|c| c.weight).sum::<f64>() / hard.len() as f64 }, // dim 4: avg hard weight
            if soft.is_empty() { 0.0 } else { soft.iter().map(|c| c.weight).sum::<f64>() / soft.len() as f64 }, // dim 5: avg soft weight
            (hard.len() as f64).max(1.0) / (constraint_count + 1.0), // dim 6: constraint balance
            domain_idx / 7.0 * complexity, // dim 7: domain-complexity interaction
        ]
    }
}

/// Compute cosine similarity between two intent vectors
pub fn intent_similarity(a: &IntentVector, b: &IntentVector) -> f64 {
    if a.dimensions.len() != b.dimensions.len() { return 0.0; }
    let dot: f64 = a.dimensions.iter().zip(b.dimensions.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.dimensions.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.dimensions.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < 0.001 || mag_b < 0.001 { return 0.0; }
    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sort_descending() {
        let parser = IntentParser::new();
        let intent = parser.parse("sort a list of numbers in descending order");
        assert_eq!(intent.domain, Domain::Sorting);
        assert!(!intent.hard_constraints.is_empty());
        assert!(intent.complexity > 0.0);
    }

    #[test]
    fn test_dimensions_length() {
        let parser = IntentParser::new();
        let intent = parser.parse("filter numbers greater than 10");
        assert_eq!(intent.dimensions.len(), 8);
    }

    #[test]
    fn test_similarity_identical() {
        let parser = IntentParser::new();
        let a = parser.parse("sort numbers descending");
        let b = parser.parse("sort numbers descending");
        let sim = intent_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_complexity_simple_vs_complex() {
        let parser = IntentParser::new();
        let simple = parser.parse("sort list");
        let complex = parser.parse("sort list of numbers in descending order and then filter duplicates and validate results");
        assert!(complex.complexity > simple.complexity);
    }
}
