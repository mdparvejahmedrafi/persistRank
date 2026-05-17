# 4. Methodology

This section presents the PersistRank framework, a duration-aware alert prioritization pipeline for e-commerce microservices. PersistRank addresses the limitations of severity-only ranking by introducing a temporal persistence signal into a composite risk computation. Given a service $v$ and time window $t$, the risk score is defined as:

$$\text{Risk}(v, t) = \alpha(v, t) \times \gamma(v) \times \pi(v, t) \tag{1}$$

where $\alpha(v, t)$ is the normalized anomaly severity, $\gamma(v)$ is the normalized business criticality, and $\pi(v, t)$ is the normalized temporal persistence score. Figure 1 illustrates this end-to-end architecture.

## 4.1 Data Acquisition and Preprocessing

We employ the RS-Anomic benchmark dataset, which contains container-level performance metrics from a 12-service Robot Shop e-commerce application. The dataset captures 19 cAdvisor resource metrics per service (228 total features) across 14 fault injection scenarios covering 10 distinct anomaly types. 

The raw metrics undergo a preprocessing pipeline that includes: 
1. **Differencing** cumulative counters into per-window rates.
2. **Missing Value Imputation** via linear interpolation.
3. **Concatenation** of normal and anomalous data into a unified, temporally aligned dataset of 123,751 monitoring windows.

## 4.2 Unsupervised Anomaly Scoring

All 228 features are standardized using z-score normalization. We employ the Isolation Forest (IF) algorithm [Liu et al., 2008] for unsupervised anomaly detection (100 estimators, 0.05 contamination). The model is trained exclusively on normal operating data. The global anomaly score is defined as the negated decision function of the IF model:

$$s(t) = -\text{IF.decision\_function}(\mathbf{x}_t) \tag{2}$$

To localize the anomaly, we compute the per-service anomaly score, $\alpha_{\text{raw}}(v_j, t)$, as the mean absolute deviation of service $v_j$'s 19 normalized features. These raw scores are then min-max normalized to produce the final severity component $\alpha(v, t) \in [0, 1]$.

## 4.3 Business Criticality Assignment

Following AlertRank-style methods, each service receives a static business criticality score $c(v) \in \{1, 2, 3, 4, 5\}$ based on its revenue impact in the checkout pipeline. The scores are normalized to $\gamma(v) \in [0, 1]$ by dividing by the maximum criticality (5).

**Table 1: Service Criticality Assignment**

| Criticality | Services | Justification |
|---|---|---|
| 5 (Critical) | payment | Direct revenue loss on failure |
| 4 (High) | cart | Checkout pipeline disruption |
| 3 (Medium) | shipping, dispatch, web, user, mongodb | Order flow / user-facing degradation |
| 2 (Low) | rabbitmq, mysql, redis | Supporting infrastructure |
| 1 (Minimal) | catalogue, ratings | Non-checkout browsing features |

## 4.4 Temporal Persistence Scoring (Core Contribution)

The persistence score captures how many consecutive time windows a service has remained anomalous. 

First, we define an anomaly threshold $\tau_j$ for each service $v_j$ as the 75th percentile of its anomaly scores during normal operations. A persistence counter $\rho(v_j, t)$ tracks consecutive anomalous windows:

$$\rho(v_j, t) = \begin{cases} \rho(v_j, t-1) + 1 & \text{if } \alpha_{\text{raw}}(v_j, t) > \tau_j \\ 0 & \text{otherwise} \end{cases} \tag{3}$$

This counter resets immediately when behavior normalizes. The raw count is normalized using a cap parameter $\kappa = 10$ to prevent unbounded growth:

$$\pi(v, t) = \min\left(\frac{\rho(v, t)}{\kappa},\ 1.0\right) \tag{4}$$

By tracking duration, PersistRank naturally amplifies slowly escalating faults (e.g., memory leaks) while attenuating transient spikes (e.g., brief CPU bursts).

## 4.5 Service Ranking and Evaluation Metric

At each window $t$, services are ranked by their composite $\text{Risk}(v, t)$ score (Eq. 1). The multiplicative nature ensures that services only receive high scores if they exhibit severe anomalies, support critical functions, *and* sustain degradation over time.

We evaluate PersistRank against five baselines: (1) AlertRank-style (Severity $\times$ Criticality), (2) No Criticality (Severity $\times$ Persistence), (3) Anomaly Only, (4) Criticality Only, and (5) Random ranking. Performance is measured using Precision@K (P@K), denoting the fraction of anomaly windows where the true affected service appears in the top-$K$ alerts.
