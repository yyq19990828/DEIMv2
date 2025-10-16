# 多卡分布式训练问题修复文档

## 问题概述

在多卡分布式训练中遇到两个主要问题:

### 问题1: Checkpoint加载时文件损坏错误
```
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

**根本原因**: 竞争条件 - 主进程正在保存checkpoint时,其他进程同时尝试加载该文件。

### 问题2: NCCL通信超时和死锁
```
RuntimeError: [../third_party/gloo/gloo/transport/tcp/unbound_buffer.cc:81] Timed out waiting 1800000ms for recv operation to complete
```

**根本原因**: 在`load_state_dict`中调用`model.state_dict()`时,DDP模型的同步操作导致不同进程处于不同状态,引发死锁。

## 已实施的修复

### 1. 在checkpoint加载前后添加分布式barrier

**文件**: `engine/solver/_solver.py`

#### 修复点1: `load_resume_state()` 方法
```python
def load_resume_state(self, path: str):
    """Load resume"""
    import os
    
    if path.startswith('http'):
        state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        # 在分布式训练中,确保所有进程同步
        if dist_utils.is_dist_available_and_initialized():
            # 确保文件存在且完整(只在rank 0检查)
            if dist_utils.get_rank() == 0:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Checkpoint file not found: {path}")
                if os.path.getsize(path) == 0:
                    raise RuntimeError(f"Checkpoint file is empty: {path}")
            
            # 同步所有进程,确保rank 0完成文件检查和保存
            torch.distributed.barrier()
        
        # 加载checkpoint
        try:
            state = torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}")
            if dist_utils.is_dist_available_and_initialized():
                print(f"Rank {dist_utils.get_rank()} failed to load checkpoint")
            raise

    self.load_state_dict(state)
    
    # 加载完成后再次同步,确保所有进程都完成加载
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.barrier()
```

#### 修复点2: `load_state_dict()` 方法中的EMA加载
```python
if k == 'ema':
    model = getattr(self, 'model', None)
    if model is not None:
        ema = dist_utils.de_parallel(v)
        # 获取去并行化的模型以避免DDP同步问题
        de_parallel_model = dist_utils.de_parallel(model)
        # 使用no_grad避免不必要的计算
        with torch.no_grad():
            model_state_dict = remove_module_prefix(de_parallel_model.state_dict())
        ema.load_state_dict({'module': model_state_dict})
        print(f'Load {k}.state_dict from model.state_dict')
```

### 2. 在训练循环中添加barrier

**文件**: `engine/solver/det_solver.py`

```python
# 第78行 - epoch开始时加载checkpoint
if epoch == self.train_dataloader.collate_fn.stop_epoch:
    # 在分布式训练中,确保所有进程同步后再加载
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.barrier()
    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

# 第160行 - 刷新EMA时加载checkpoint
elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
    best_stat = {'epoch': -1, }
    self.ema.decay -= 0.0001
    # 在分布式训练中,确保所有进程同步后再加载
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.barrier()
    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
```

## 如果问题仍然存在

如果在应用上述修复后仍然遇到NCCL超时问题,可以尝试以下额外措施:

### 1. 增加NCCL超时时间

在训练脚本开始处添加:
```python
import os
# 将超时时间从默认的30分钟增加到60分钟
os.environ['NCCL_TIMEOUT'] = '3600'  # 秒
```

### 2. 启用NCCL调试日志

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 3. 使用Gloo后端进行CPU通信

在某些情况下,可以尝试使用Gloo后端:
```python
# 在 engine/misc/dist_utils.py 的 setup_distributed 函数中
torch.distributed.init_process_group(backend='gloo', init_method='env://')
```

### 4. 禁用EMA的自动刷新

如果问题主要出现在EMA刷新时,可以临时禁用自动刷新:

在配置文件中设置:
```yaml
# 将stop_epoch设置为一个很大的值,避免触发EMA刷新
stop_epoch: 999999
```

### 5. 检查checkpoint文件完整性

添加更严格的文件完整性检查:
```python
import zipfile

def verify_checkpoint(path):
    """验证checkpoint文件是否完整"""
    try:
        # PyTorch checkpoint是zip格式
        with zipfile.ZipFile(path, 'r') as zf:
            # 测试zip文件完整性
            bad_file = zf.testzip()
            if bad_file is not None:
                raise RuntimeError(f"Corrupted file in checkpoint: {bad_file}")
        return True
    except Exception as e:
        print(f"Checkpoint verification failed: {e}")
        return False
```

### 6. 使用文件锁机制

对于频繁保存和加载的场景,可以使用文件锁:
```python
import fcntl
import time

def safe_torch_save(obj, path):
    """使用文件锁安全保存checkpoint"""
    temp_path = f"{path}.tmp"
    torch.save(obj, temp_path)
    
    # 原子性重命名
    os.replace(temp_path, path)
    
    # 确保文件系统同步
    with open(path, 'rb') as f:
        os.fsync(f.fileno())
```

## 监控和调试建议

### 1. 添加详细的日志

在关键位置添加日志:
```python
if dist_utils.is_dist_available_and_initialized():
    rank = dist_utils.get_rank()
    print(f"[Rank {rank}] Before barrier at {time.time()}")
    torch.distributed.barrier()
    print(f"[Rank {rank}] After barrier at {time.time()}")
```

### 2. 使用PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # 训练代码
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 3. 检查GPU内存使用

```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 最佳实践

1. **定期清理checkpoint**: 只保留最近的几个checkpoint,避免磁盘空间问题
2. **使用SSD存储checkpoint**: 提高I/O性能
3. **避免在训练循环中频繁保存**: 使用合理的保存间隔
4. **监控网络带宽**: 确保节点间通信正常
5. **使用环境变量控制行为**: 便于调试和配置

## 相关环境变量

```bash
# NCCL相关
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# PyTorch分布式
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0
```

## 总结

本次修复主要解决了:
1. ✅ Checkpoint文件竞争条件问题
2. ✅ 分布式训练中的同步问题
3. ✅ EMA加载时的DDP同步问题
4. ✅ 添加了完整的错误处理和日志

如果问题持续存在,请检查:
- 网络连接是否稳定
- GPU是否正常工作
- 磁盘I/O是否正常
- 是否有足够的内存和显存