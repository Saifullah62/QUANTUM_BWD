# Crucible Commercial Notes

Licensing, enterprise features, and deployment considerations.

## License Structure

### Open Source (Apache 2.0)

The core training and evaluation pipeline is open source:

- **Bundle Generator**: Tiered contrastive bundle creation
- **Curriculum Sampler**: Multi-tier sampling with ratio controls
- **Training Loop**: Standard contrastive learning training
- **Evaluation Engine**: Scoreboard generation, margin metrics
- **Report Generator**: Basic visualization and reporting
- **CLI**: All standard commands (train, eval, report)

### Commercial License

Advanced features require a commercial license:

- **Organic Miner**: Multi-lineup validation mining algorithm
- **Tier3 Expansion**: Automated killer generation with quality gating
- **Enterprise Guardrails**: Advanced dilution detection, regression alerting
- **Pareto Optimizer**: Automated sweep orchestration and analysis
- **Team Features**: Multi-user dashboards, experiment tracking integration

## Feature Comparison

| Feature | Open Source | Commercial |
|---------|-------------|------------|
| Bundle generation | Yes | Yes |
| Curriculum training | Yes | Yes |
| Basic evaluation | Yes | Yes |
| Danger scoring | Yes | Yes |
| Margin metrics | Yes | Yes |
| CLI commands | Yes | Yes |
| Organic mining | - | Yes |
| Multi-lineup validation | - | Yes |
| Tier3 expansion | - | Yes |
| Pareto frontier analysis | Basic | Advanced |
| Automated sweeps | Manual | Orchestrated |
| Dilution guardrails | Warning only | Block + notify |
| SSO/RBAC | - | Yes |
| MLflow/W&B integration | - | Yes |
| Priority support | Community | Dedicated |

## Deployment Options

### Self-Hosted

Run Crucible on your own infrastructure:

```bash
pip install crucible-ml

# Or with commercial features
pip install crucible-ml[enterprise]
export CRUCIBLE_LICENSE_KEY="your-key-here"
```

**Requirements**:
- Python 3.9+
- CUDA GPU (recommended for training)
- 8GB+ GPU memory
- License key for enterprise features

### Cloud Deployment

Containerized deployment for cloud environments:

```dockerfile
FROM python:3.10-slim
RUN pip install crucible-ml[enterprise]
COPY crucible.yaml /app/config.yaml
WORKDIR /app
ENTRYPOINT ["crucible"]
```

**Supported Platforms**:
- AWS (EC2, ECS, EKS)
- GCP (Compute Engine, GKE)
- Azure (VMs, AKS)
- On-premise Kubernetes

### Managed Service

Fully managed Crucible as a service (roadmap):

- No infrastructure management
- Automatic scaling
- Built-in experiment tracking
- Team collaboration features
- SLA-backed availability

## Enterprise Features Detail

### Organic Miner

The organic mining algorithm with multi-lineup validation is a validated mechanism that prevents pseudo-killer contamination. Core innovations:

1. **Multi-Lineup Validation**: Test candidates against multiple independent negative lineups
2. **Pass Threshold Gating**: Require minimum passes (e.g., 3/10) for acceptance
3. **Quality Stratification**: Track pass counts for quality-aware sampling

**Value**: Teams report 30-50% improvement in generalization metrics when using properly validated organic killers vs naive hard negative mining.

### Enterprise Guardrails

Beyond open-source warnings:

- **Blocking Guardrails**: Prevent training with dangerous configurations
- **Regression Alerting**: Slack/PagerDuty integration for metric drops
- **Audit Logging**: Full history of config changes and experiment runs
- **Approval Workflows**: Require sign-off before production deployments

### Team Collaboration

- **Experiment Dashboard**: Web UI for tracking sweeps and results
- **Capsule Registry**: Centralized storage for reproducible artifacts
- **Role-Based Access**: Separate researcher/reviewer/admin permissions
- **Integration APIs**: Connect to existing MLOps infrastructure

## Pricing Tiers

### Community (Free)

- Open source features only
- Community support via GitHub
- Unlimited local usage

### Team ($X/month per seat)

- All commercial features
- Email support
- 5+ seats minimum
- Annual billing

### Enterprise (Custom)

- All features
- Dedicated support engineer
- Custom integrations
- On-premise deployment
- SLA guarantees
- Training and onboarding

Contact: sales@crucible-ml.com

## Integration Ecosystem

### Experiment Tracking

```python
# MLflow integration
crucible train --config config.yaml --tracking mlflow

# Weights & Biases integration
crucible train --config config.yaml --tracking wandb
```

### Model Registry

```python
# Register checkpoint with metadata
crucible capsule register \
  --capsule baseline_v1.tar.gz \
  --registry mlflow \
  --tags production,validated
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run Crucible validation
  run: |
    crucible train --config ci_config.yaml --steps 1000
    crucible eval --checkpoint checkpoint.pt --eval-set eval.jsonl
    crucible guardrail check --baseline baseline_scoreboard.json
```

## Data Privacy

### On-Premise Guarantee

Crucible runs entirely on your infrastructure. No data leaves your environment:

- Training data stays local
- Embeddings computed locally
- No cloud dependencies for core features
- License validation via offline key (enterprise)

### Telemetry (Optional)

Anonymous usage telemetry can be enabled to improve the product:

```yaml
telemetry:
  enabled: false  # Default: off
  anonymous: true
  metrics_only: true  # No data content
```

Telemetry includes:
- Feature usage counts
- Error rates
- Performance metrics

Telemetry **never** includes:
- Training data
- Model weights
- Eval results
- File contents

## Support

### Community Support

- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, best practices
- Documentation: This repository

### Commercial Support

- Email: support@crucible-ml.com
- Response SLA: 24h (Team), 4h (Enterprise)
- Dedicated Slack channel (Enterprise)
- Quarterly business reviews (Enterprise)

## Roadmap

### Current (v1.0)

- Core training/eval pipeline
- CLI with standard commands
- Basic guardrails
- Capsule packaging

### Near-Term (v1.1)

- Web dashboard (Enterprise)
- MLflow/W&B integration
- Automated sweep orchestration
- Enhanced reporting

### Future

- Managed cloud service
- Multi-model support (beyond contrastive)
- AutoML for curriculum optimization
- Distributed training support

## FAQ

**Q: Can I use the open source version commercially?**
A: Yes, Apache 2.0 allows commercial use. You cannot use the organic miner or enterprise features without a commercial license.

**Q: What's the minimum viable usage of Crucible?**
A: Open source Crucible provides complete training and evaluation. You can build effective contrastive models without commercial features. The organic miner and advanced guardrails accelerate iteration but aren't required.

**Q: Can I contribute to the open source project?**
A: Yes! Contributions to the Apache 2.0 licensed components are welcome. See CONTRIBUTING.md for guidelines.

**Q: How does licensing work for the organic miner I develop myself?**
A: If you implement your own mining algorithm, you own it. The commercial license covers our specific multi-lineup validation implementation.

**Q: Is there a trial for commercial features?**
A: Yes, we offer 30-day trials for Team tier. Contact sales@crucible-ml.com.
