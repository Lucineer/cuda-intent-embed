//! # cuda-intent-embed
//!
//! Intent embedding system — converts natural language intents into
//! 8-dimensional confidence vectors. Agents compare intents via cosine similarity.
//!
//! ```rust
//! use cuda_intent_embed::{IntentParser, Domain, ConstraintKind};
//! use cuda_equipment::Confidence;
//!
//! let parser = IntentParser::new();
//! let vec = parser.parse("navigate to checkpoint alpha", Confidence::SURE);
//! assert!(vec.domain == Some(Domain::Navigation));
//! ```

pub use cuda_equipment::Confidence;

use std::collections::HashMap;

/// Known domains an agent can operate in.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Domain {
    Navigation,
    Communication,
    Computation,
    Sensing,
    Actuation,
    Coordination,
    Learning,
    Maintenance,
    Exploration,
    Defense,
    Creative,
    Analysis,
}

impl Domain {
    pub fn all() -> Vec<Domain> {
        vec![Domain::Navigation, Domain::Communication, Domain::Computation,
            Domain::Sensing, Domain::Actuation, Domain::Coordination,
            Domain::Learning, Domain::Maintenance, Domain::Exploration,
            Domain::Defense, Domain::Creative, Domain::Analysis]
    }

    pub fn as_str(&self) -> &str {
        match self {
            Domain::Navigation => "navigation", Domain::Communication => "communication",
            Domain::Computation => "computation", Domain::Sensing => "sensing",
            Domain::Actuation => "actuation", Domain::Coordination => "coordination",
            Domain::Learning => "learning", Domain::Maintenance => "maintenance",
            Domain::Exploration => "exploration", Domain::Defense => "defense",
            Domain::Creative => "creative", Domain::Analysis => "analysis",
        }
    }
}

/// Types of constraints on an intent.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    MaxResources { tokens: u64, memory_mb: u64 },
    Timeout { seconds: u64 },
    Precision { bits: u8 },
    Safety { level: u8 }, // 0=none, 1=safe, 2=verified
    Privacy { anonymize: bool },
    Determinism { required: bool },
}

/// An embedded intent — the core data structure.
#[derive(Debug, Clone)]
pub struct IntentVector {
    pub values: [f64; 8],     // 8-dimensional embedding
    pub confidence: Confidence,
    pub domain: Option<Domain>,
    pub constraints: Vec<ConstraintKind>,
    pub raw_text: String,
    pub source_agent: Option<u64>,
    pub timestamp: u64,
}

impl IntentVector {
    pub fn zero() -> Self {
        Self { values: [0.0; 8], confidence: Confidence::ZERO,
            domain: None, constraints: vec![], raw_text: String::new(),
            source_agent: None, timestamp: 0 }
    }

    /// Cosine similarity with another intent vector.
    pub fn similarity(&self, other: &IntentVector) -> f64 {
        let mut dot = 0.0f64;
        let mut mag_a = 0.0f64;
        let mut mag_b = 0.0f64;
        for i in 0..8 {
            dot += self.values[i] * other.values[i];
            mag_a += self.values[i] * self.values[i];
            mag_b += other.values[i] * other.values[i];
        }
        let denom = mag_a.sqrt() * mag_b.sqrt();
        if denom < 1e-10 { return 0.0; }
        dot / denom
    }

    /// Weighted similarity combining vector similarity with confidence.
    pub fn weighted_similarity(&self, other: &IntentVector) -> f64 {
        let vec_sim = self.similarity(other);
        let conf_sim = self.confidence.combine(other.confidence).value();
        vec_sim * 0.7 + conf_sim * 0.3
    }

    /// Euclidean distance between intent vectors.
    pub fn distance(&self, other: &IntentVector) -> f64 {
        let sum: f64 = (0..8).map(|i| (self.values[i] - other.values[i]).powi(2)).sum();
        sum.sqrt()
    }

    /// Average of two intent vectors (intent averaging).
    pub fn average(&self, other: &IntentVector) -> IntentVector {
        let values: [f64; 8] = std::array::from_fn(|i| (self.values[i] + other.values[i]) / 2.0);
        IntentVector { values, confidence: self.confidence.combine(other.confidence),
            domain: self.domain.clone().or(other.domain.clone()),
            constraints: vec![], raw_text: String::new(),
            source_agent: None, timestamp: now_ms() }
    }
}

/// Parses natural language into intent vectors.
pub struct IntentParser {
    domain_keywords: HashMap<Domain, Vec<String>>,
    domain_weights: [f64; 8], // weight vector per dimension for each domain
}

impl IntentParser {
    pub fn new() -> Self {
        let mut keywords = HashMap::new();
        keywords.insert(Domain::Navigation, vec!["navigate".into(), "go".into(), "move".into(),
            "path".into(), "route".into(), "waypoint".into(), "checkpoint".into(),
            "location".into(), "position".into(), "heading".into(), "travel".into()]);
        keywords.insert(Domain::Communication, vec!["send".into(), "message".into(), "tell".into(),
            "notify".into(), "broadcast".into(), "signal".into(), "report".into(), "alert".into()]);
        keywords.insert(Domain::Computation, vec!["compute".into(), "calculate".into(), "process".into(),
            "analyze".into(), "compile".into(), "optimize".into(), "transform".into(), "render".into()]);
        keywords.insert(Domain::Sensing, vec!["observe".into(), "detect".into(), "measure".into(),
            "scan".into(), "read".into(), "monitor".into(), "perceive".into(), "sense".into()]);
        keywords.insert(Domain::Actuation, vec!["actuate".into(), "move".into(), "grasp".into(),
            "deploy".into(), "activate".into(), "release".into(), "push".into(), "pull".into()]);
        keywords.insert(Domain::Coordination, vec!["coordinate".into(), "assign".into(), "delegate".into(),
            "schedule".into(), "sync".into(), "cooperate".into(), "team".into(), "share".into()]);
        keywords.insert(Domain::Learning, vec!["learn".into(), "train".into(), "adapt".into(),
            "improve".into(), "update".into(), "evolve".into(), "memorize".into(), "recall".into()]);
        keywords.insert(Domain::Exploration, vec!["explore".into(), "discover".into(), "search".into(),
            "investigate".into(), "survey".into(), "map".into(), "scan_area".into(), "probe".into()]);
        keywords.insert(Domain::Defense, vec!["defend".into(), "protect".into(), "shield".into(),
            "block".into(), "alert".into(), "secure".into(), "guard".into(), "verify".into()]);
        keywords.insert(Domain::Creative, vec!["create".into(), "design".into(), "generate".into(),
            "imagine".into(), "compose".into(), "invent".into(), "craft".into(), "build".into()]);
        keywords.insert(Domain::Analysis, vec!["analyze".into(), "evaluate".into(), "assess".into(),
            "compare".into(), "rank".into(), "score".into(), "diagnose".into(), "audit".into()]);
        keywords.insert(Domain::Maintenance, vec!["repair".into(), "fix".into(), "maintain".into(),
            "service".into(), "calibrate".into(), "refuel".into(), "charge".into(), "update_sys".into()]);

        // Each domain maps to a unique 8-dim vector for differentiation
        let weights: HashMap<Domain, [f64; 8]> = [
            (Domain::Navigation, [0.9, 0.3, 0.1, 0.8, 0.2, 0.1, 0.1, 0.2]),
            (Domain::Communication, [0.2, 0.9, 0.3, 0.1, 0.7, 0.3, 0.1, 0.1]),
            (Domain::Computation, [0.1, 0.3, 0.9, 0.2, 0.1, 0.8, 0.3, 0.1]),
            (Domain::Sensing, [0.3, 0.1, 0.2, 0.9, 0.1, 0.1, 0.8, 0.3]),
            (Domain::Actuation, [0.8, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2, 0.9]),
            (Domain::Coordination, [0.3, 0.7, 0.2, 0.1, 0.9, 0.3, 0.1, 0.2]),
            (Domain::Learning, [0.1, 0.2, 0.3, 0.2, 0.3, 0.9, 0.7, 0.1]),
            (Domain::Exploration, [0.7, 0.2, 0.1, 0.8, 0.2, 0.1, 0.1, 0.7]),
            (Domain::Defense, [0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.3, 0.5]),
            (Domain::Creative, [0.6, 0.3, 0.8, 0.1, 0.2, 0.2, 0.5, 0.7]),
            (Domain::Analysis, [0.2, 0.3, 0.7, 0.3, 0.3, 0.7, 0.5, 0.2]),
            (Domain::Maintenance, [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
        ].into_iter().collect();

        Self { domain_keywords: keywords, domain_weights: [0.0; 8] }
    }

    /// Parse text into an intent vector.
    pub fn parse(&self, text: &str, confidence: Confidence) -> IntentVector {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();

        // Find best matching domain
        let mut best_domain: Option<Domain> = None;
        let mut best_score = 0usize;

        for (domain, keywords) in &self.domain_keywords {
            let score = keywords.iter().filter(|kw| words.iter().any(|w| w.starts_with(kw.as_str()) || kw.as_str().contains(w))).count();
            if score > best_score {
                best_score = score;
                best_domain = Some(domain.clone());
            }
        }

        // Generate embedding based on domain and text features
        let values = self.embed(&text_lower, &best_domain);

        IntentVector { values, confidence, domain: best_domain,
            constraints: vec![], raw_text: text.to_string(),
            source_agent: None, timestamp: now_ms() }
    }

    fn embed(&self, text: &str, domain: &Option<Domain>) -> [f64; 8] {
        // Dimension meanings:
        // 0=urgency, 1=social, 2=complexity, 3=spatial, 4=collaboration, 5=precision, 6=perception, 7=action
        let urgency_words = ["urgent", "now", "immediately", "critical", "emergency", "asap"];
        let social_words = ["team", "share", "tell", "notify", "broadcast", "cooperate"];
        let complex_words = ["complex", "multi", "analyze", "evaluate", "optimize", "compile"];
        let spatial_words = ["location", "position", "coordinate", "grid", "map", "area"];
        let collab_words = ["coordinate", "sync", "together", "assign", "delegate", "help"];
        let precision_words = ["exact", "precise", "carefully", "verify", "check", "measure"];
        let percept_words = ["see", "detect", "observe", "sense", "monitor", "read"];

        let word_set: Vec<&str> = text.split_whitespace().collect();
        let count = |list: &[&str]| list.iter().filter(|w| word_set.iter().any(|ws| ws.starts_with(*w))).count();
        let norm = |c| (c as f64 / 3.0).min(1.0);

        let mut values = [0.5; 8]; // baseline
        values[0] = norm(count(&urgency_words));
        values[1] = norm(count(&social_words));
        values[2] = norm(count(&complex_words));
        values[3] = norm(count(&spatial_words));
        values[4] = norm(count(&collab_words));
        values[5] = norm(count(&precision_words));
        values[6] = norm(count(&percept_words));
        values[7] = 1.0 - values[6]; // action ≈ inverse of perception

        // Blend with domain signature
        if let Some(d) = domain {
            let domain_vec = self.domain_vector(d);
            for i in 0..8 { values[i] = values[i] * 0.5 + domain_vec[i] * 0.5; }
        }

        values
    }

    fn domain_vector(&self, domain: &Domain) -> [f64; 8] {
        match domain {
            Domain::Navigation => [0.9, 0.3, 0.1, 0.8, 0.2, 0.1, 0.1, 0.2],
            Domain::Communication => [0.2, 0.9, 0.3, 0.1, 0.7, 0.3, 0.1, 0.1],
            Domain::Computation => [0.1, 0.3, 0.9, 0.2, 0.1, 0.8, 0.3, 0.1],
            Domain::Sensing => [0.3, 0.1, 0.2, 0.9, 0.1, 0.1, 0.8, 0.3],
            Domain::Actuation => [0.8, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2, 0.9],
            Domain::Coordination => [0.3, 0.7, 0.2, 0.1, 0.9, 0.3, 0.1, 0.2],
            Domain::Learning => [0.1, 0.2, 0.3, 0.2, 0.3, 0.9, 0.7, 0.1],
            Domain::Exploration => [0.7, 0.2, 0.1, 0.8, 0.2, 0.1, 0.1, 0.7],
            Domain::Defense => [0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.3, 0.5],
            Domain::Creative => [0.6, 0.3, 0.8, 0.1, 0.2, 0.2, 0.5, 0.7],
            Domain::Analysis => [0.2, 0.3, 0.7, 0.3, 0.3, 0.7, 0.5, 0.2],
            Domain::Maintenance => [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        }
    }

    /// Find the most similar domain for an intent.
    pub fn classify(&self, intent: &IntentVector) -> Option<Domain> {
        let mut best: Option<Domain> = None;
        let mut best_sim = 0.0f64;
        for domain in Domain::all() {
            let domain_vec = self.domain_vector(&domain);
            let probe = IntentVector { values: domain_vec, confidence: Confidence::SURE,
                domain: Some(domain.clone()), constraints: vec![], raw_text: String::new(),
                source_agent: None, timestamp: 0 };
            let sim = intent.similarity(&probe);
            if sim > best_sim { best_sim = sim; best = Some(domain); }
        }
        best
    }
}

impl Default for IntentParser { fn default() -> Self { Self::new() } }

/// Intent cache — stores and retrieves recent intents.
pub struct IntentCache {
    entries: Vec<(String, IntentVector)>,
    max_size: usize,
}

impl IntentCache {
    pub fn new(max_size: usize) -> Self { Self { entries: vec![], max_size } }

    pub fn store(&mut self, text: &str, intent: IntentVector) {
        if self.entries.len() >= self.max_size { self.entries.remove(0); }
        self.entries.push((text.to_string(), intent));
    }

    /// Find most similar cached intent.
    pub fn find_similar(&self, intent: &IntentVector, threshold: f64) -> Option<(&str, f64)> {
        let mut best: Option<(&str, f64)> = None;
        for (text, cached) in &self.entries {
            let sim = intent.similarity(cached);
            if sim > threshold {
                best = Some((text.as_str(), best.map_or(sim, |b| sim.max(b.1))));
            }
        }
        best
    }

    pub fn len(&self) -> usize { self.entries.len() }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_navigation() {
        let parser = IntentParser::new();
        let v = parser.parse("navigate to checkpoint alpha", Confidence::SURE);
        assert_eq!(v.domain, Some(Domain::Navigation));
    }

    #[test]
    fn test_parse_communication() {
        let parser = IntentParser::new();
        let v = parser.parse("send message to team", Confidence::SURE);
        assert_eq!(v.domain, Some(Domain::Communication));
    }

    #[test]
    fn test_parse_computation() {
        let parser = IntentParser::new();
        let v = parser.parse("optimize the compiler pipeline", Confidence::SURE);
        assert_eq!(v.domain, Some(Domain::Computation));
    }

    #[test]
    fn test_similarity_same_domain() {
        let parser = IntentParser::new();
        let a = parser.parse("navigate to waypoint", Confidence::SURE);
        let b = parser.parse("go to location", Confidence::SURE);
        assert!(a.similarity(&b) > 0.5);
    }

    #[test]
    fn test_similarity_different_domain() {
        let parser = IntentParser::new();
        let a = parser.parse("navigate forward", Confidence::SURE);
        let b = parser.parse("send alert now", Confidence::SURE);
        let sim = a.similarity(&b);
        // Different domains should have lower similarity
        assert!(sim < a.similarity(&a)); // same intent > cross-domain
    }

    #[test]
    fn test_zero_vector() {
        let z = IntentVector::zero();
        assert_eq!(z.confidence, Confidence::ZERO);
    }

    #[test]
    fn test_distance() {
        let a = IntentVector { values: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            confidence: Confidence::SURE, domain: None, constraints: vec![],
            raw_text: String::new(), source_agent: None, timestamp: 0 };
        let b = IntentVector { values: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            confidence: Confidence::SURE, domain: None, constraints: vec![],
            raw_text: String::new(), source_agent: None, timestamp: 0 };
        assert!((a.distance(&b) - 1.414).abs() < 0.01); // sqrt(2)
    }

    #[test]
    fn test_classify() {
        let parser = IntentParser::new();
        let v = parser.parse("observe the sensor readings", Confidence::SURE);
        let classified = parser.classify(&v);
        assert!(classified.is_some());
    }

    #[test]
    fn test_cache() {
        let mut cache = IntentCache::new(10);
        let parser = IntentParser::new();
        let v = parser.parse("navigate forward", Confidence::SURE);
        cache.store("nav", v.clone());
        let found = cache.find_similar(&v, 0.5);
        assert!(found.is_some());
        assert_eq!(found.unwrap().0, "nav");
    }

    #[test]
    fn test_average() {
        let a = IntentVector { values: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            confidence: Confidence::SURE, domain: Some(Domain::Navigation),
            constraints: vec![], raw_text: String::new(), source_agent: None, timestamp: 0 };
        let b = IntentVector { values: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            confidence: Confidence::SURE, domain: Some(Domain::Communication),
            constraints: vec![], raw_text: String::new(), source_agent: None, timestamp: 0 };
        let avg = a.average(&b);
        assert!((avg.values[0] - 0.5).abs() < 0.01);
        assert!((avg.values[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_urgency_detection() {
        let parser = IntentParser::new();
        let urgent = parser.parse("urgent emergency now critical", Confidence::SURE);
        let calm = parser.parse("eventually maybe consider", Confidence::SURE);
        assert!(urgent.values[0] > calm.values[0]); // urgency dim
    }
}
