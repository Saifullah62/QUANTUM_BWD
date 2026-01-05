"""
GPU Cluster Management
======================

Utilities for coordinating training across the GPU cluster:
- Local: RTX 5070 Ti (16GB)
- gpu-swarm: RTX 4000 Ada (20GB) - Fleet services
- gpu-ramp: RTX 6000 Ada (48GB) - Large model training

Total: 84GB VRAM
"""

import asyncio
import httpx
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class GPURole(Enum):
    """Role assignment for each GPU node"""
    INFERENCE = "inference"      # Running inference for data generation
    TRAINING = "training"        # Training the QLLM model
    EVALUATION = "evaluation"    # Running evaluation benchmarks
    SWARM = "swarm"             # Multi-agent swarm processing


@dataclass
class GPUNode:
    """Represents a GPU node in the cluster"""
    name: str
    host: str
    gpu_model: str
    vram_gb: int
    role: GPURole
    ssh_alias: Optional[str] = None
    ollama_models: List[str] = None

    def __post_init__(self):
        if self.ollama_models is None:
            self.ollama_models = []


class ClusterManager:
    """
    Manage the GPU cluster for QLLM training and inference.

    Architecture:
    - gpu-swarm (20GB): Runs Fleet services, handles data generation via swarm
    - gpu-ramp (48GB): Primary training node for large models
    - Local (16GB): Coordination, evaluation, small experiments
    """

    # Cluster configuration
    NODES = {
        'local': GPUNode(
            name='local',
            host='localhost',
            gpu_model='RTX 5070 Ti',
            vram_gb=16,
            role=GPURole.EVALUATION
        ),
        'gpu-swarm': GPUNode(
            name='gpu-swarm',
            host='159.203.35.45',
            gpu_model='RTX 4000 Ada',
            vram_gb=20,
            role=GPURole.SWARM,
            ssh_alias='gpu-swarm',
            ollama_models=['mixtral:8x7b', 'codellama:13b', 'llama3.1:8b',
                          'llama3.2:3b', 'phi3:mini', 'nomic-embed-text']
        ),
        'gpu-ramp': GPUNode(
            name='gpu-ramp',
            host='159.89.127.151',
            gpu_model='RTX 6000 Ada',
            vram_gb=48,
            role=GPURole.TRAINING,
            ssh_alias='gpu-ramp'
        )
    }

    # Fleet service ports on gpu-swarm
    FLEET_SERVICES = {
        'api_gateway': 8000,
        'model_router': 8001,
        'context_manager': 8002,
        'task_queue': 8003,
        'result_aggregator': 8004,
        'health_monitor': 8005,
        'metrics_collector': 8006,
        'swarm_controller': 8007,
        'experiment_tracker': 8008,
        'data_pipeline': 8009,
        'model_registry': 8010,
        'inference_cache': 8011
    }

    def __init__(self):
        self.swarm_url = f"http://{self.NODES['gpu-swarm'].host}:8007"

    async def check_node_health(self, node_name: str) -> Dict[str, Any]:
        """Check health of a specific node"""
        node = self.NODES.get(node_name)
        if not node:
            return {'status': 'unknown', 'error': f'Unknown node: {node_name}'}

        if node_name == 'local':
            # Check local GPU
            try:
                import torch
                if torch.cuda.is_available():
                    return {
                        'status': 'healthy',
                        'gpu': torch.cuda.get_device_name(0),
                        'vram_free': torch.cuda.mem_get_info()[0] / 1e9
                    }
                else:
                    return {'status': 'degraded', 'error': 'CUDA not available'}
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        else:
            # Check remote node via SSH
            try:
                result = subprocess.run(
                    ['ssh', node.ssh_alias, 'nvidia-smi', '--query-gpu=name,memory.free',
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return {
                        'status': 'healthy',
                        'gpu_info': result.stdout.strip()
                    }
                else:
                    return {'status': 'error', 'error': result.stderr}
            except Exception as e:
                return {'status': 'error', 'error': str(e)}

    async def check_fleet_services(self) -> Dict[str, Any]:
        """Check status of Fleet services on gpu-swarm"""
        results = {}
        async with httpx.AsyncClient(timeout=5) as client:
            for service, port in self.FLEET_SERVICES.items():
                try:
                    url = f"http://{self.NODES['gpu-swarm'].host}:{port}/health"
                    response = await client.get(url)
                    results[service] = {
                        'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                        'port': port
                    }
                except Exception as e:
                    results[service] = {
                        'status': 'unreachable',
                        'port': port,
                        'error': str(e)
                    }
        return results

    async def call_swarm(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: float = 120
    ) -> Dict[str, Any]:
        """Call the swarm controller API"""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.swarm_url}{endpoint}",
                json=data
            )
            return response.json()

    async def swarm_think(self, problem: str) -> Dict[str, Any]:
        """Use swarm for thinking/reasoning"""
        return await self.call_swarm('/swarm/think', {'problem': problem})

    async def swarm_explore(
        self,
        problem: str,
        num_explorers: int = 3
    ) -> Dict[str, Any]:
        """Use swarm for parallel exploration"""
        return await self.call_swarm('/swarm/explore', {
            'problem': problem,
            'num_explorers': num_explorers
        })

    async def swarm_debate(
        self,
        topic: str,
        positions: List[str]
    ) -> Dict[str, Any]:
        """Use swarm for debate between positions"""
        return await self.call_swarm('/swarm/debate', {
            'topic': topic,
            'positions': positions
        })

    def get_training_node(self) -> GPUNode:
        """Get the primary training node (gpu-ramp with 48GB)"""
        return self.NODES['gpu-ramp']

    def get_inference_node(self) -> GPUNode:
        """Get the inference node for data generation"""
        return self.NODES['gpu-swarm']

    def prepare_training_command(
        self,
        script_path: str,
        args: Dict[str, Any]
    ) -> str:
        """Prepare SSH command to run training on gpu-ramp"""
        node = self.get_training_node()

        # Build argument string
        arg_str = ' '.join(f'--{k}={v}' for k, v in args.items())

        # Full command
        cmd = f"ssh {node.ssh_alias} 'cd /workspace && python {script_path} {arg_str}'"
        return cmd

    def sync_code_to_node(self, node_name: str, local_path: str, remote_path: str):
        """Sync code to a remote node"""
        node = self.NODES.get(node_name)
        if not node or not node.ssh_alias:
            raise ValueError(f"Invalid node or no SSH alias: {node_name}")

        cmd = f"rsync -avz {local_path}/ {node.ssh_alias}:{remote_path}/"
        subprocess.run(cmd, shell=True, check=True)

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get full cluster status"""
        status = {
            'nodes': {},
            'total_vram_gb': sum(n.vram_gb for n in self.NODES.values()),
            'fleet_services': await self.check_fleet_services()
        }

        for name in self.NODES:
            status['nodes'][name] = await self.check_node_health(name)

        return status


class DistributedTrainer:
    """
    Coordinate distributed training across the cluster.

    Strategy:
    - Data generation: gpu-swarm (swarm controller + ollama models)
    - Training: gpu-ramp (48GB VRAM for large models)
    - Evaluation: local (quick iteration)
    """

    def __init__(self, cluster: ClusterManager):
        self.cluster = cluster

    async def generate_data_batch(
        self,
        paradigm: str,
        num_examples: int
    ) -> List[Dict]:
        """Generate training data using swarm"""
        prompts_by_paradigm = {
            'semantic_phase': "Generate an example showing how the same word has different meanings in different contexts",
            'retrocausal': "Generate an example of reasoning backwards from an outcome to its causes",
            'lindblad': "Generate an example of finding stable patterns in noisy, chaotic information",
            'qualia': "Generate an example describing the subjective, qualitative experience of something",
            'emergent': "Generate an example of emergent properties arising from complex systems"
        }

        prompt = prompts_by_paradigm.get(paradigm, prompts_by_paradigm['semantic_phase'])

        results = []
        for _ in range(num_examples):
            result = await self.cluster.swarm_think(prompt)
            if result.get('status') == 'SUCCESS':
                results.append({
                    'input': prompt,
                    'output': result.get('answer', ''),
                    'paradigm': paradigm
                })

        return results

    def launch_training_job(
        self,
        config_path: str,
        data_path: str,
        output_dir: str
    ) -> subprocess.Popen:
        """Launch training job on gpu-ramp"""
        cmd = self.cluster.prepare_training_command(
            'qllm/training/trainer.py',
            {
                'config': config_path,
                'data': data_path,
                'output-dir': output_dir
            }
        )

        return subprocess.Popen(cmd, shell=True)


# CLI for cluster management
async def main():
    """Cluster management CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="QLLM Cluster Manager")
    parser.add_argument('--status', action='store_true', help='Show cluster status')
    parser.add_argument('--sync', type=str, help='Sync code to node')
    parser.add_argument('--test-swarm', action='store_true', help='Test swarm connection')

    args = parser.parse_args()

    cluster = ClusterManager()

    if args.status:
        status = await cluster.get_cluster_status()
        import json
        print(json.dumps(status, indent=2))

    elif args.test_swarm:
        result = await cluster.swarm_think("What are the implications of quantum semantics for language models?")
        print(f"Swarm response: {result}")

    elif args.sync:
        cluster.sync_code_to_node(args.sync, '.', '/workspace/qllm')
        print(f"Code synced to {args.sync}")


if __name__ == "__main__":
    asyncio.run(main())
