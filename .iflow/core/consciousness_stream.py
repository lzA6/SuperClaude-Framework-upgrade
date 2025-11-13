#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 意识流系统 (Consciousness Stream System)
Global Consciousness Stream - 记录、分析、预测、进化的全局意识流系统

实现跨智能体的全局意识流，具备长期记忆、模式识别、预测能力和自进化功能。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading
import time

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessEvent:
    """意识流事件"""
    event_id: str
    timestamp: datetime
    event_type: str
    agent_id: str
    context: Dict[str, Any]
    outcome: Any
    semantic_vector: np.ndarray
    emotional_weight: float
    importance_score: float
    related_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryPattern:
    """记忆模式"""
    pattern_id: str
    pattern_type: str
    frequency: int
    success_rate: float
    context_signature: np.ndarray
    outcome_prediction: Any
    last_seen: datetime
    confidence: float

class KnowledgeGraph:
    """知识图谱 - 长期记忆存储"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or Path.cwd() / ".iflow" / "knowledge"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.entities = {}  # 实体字典
        self.relations = defaultdict(list)  # 关系字典
        self.embeddings = {}  # 向量嵌入
        
        self._load_knowledge_graph()
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]):
        """添加实体"""
        self.entities[entity_id] = {
            'type': entity_type,
            'properties': properties,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        self._save_knowledge_graph()
    
    def add_relation(self, subject: str, predicate: str, obj: str, confidence: float = 1.0):
        """添加关系"""
        relation_id = f"{subject}_{predicate}_{obj}"
        self.relations[subject].append({
            'relation_id': relation_id,
            'predicate': predicate,
            'object': obj,
            'confidence': confidence,
            'created_at': datetime.now()
        })
        self._save_knowledge_graph()
    
    def query_relations(self, entity: str, predicate: Optional[str] = None) -> List[Dict]:
        """查询关系"""
        relations = self.relations.get(entity, [])
        if predicate:
            relations = [r for r in relations if r['predicate'] == predicate]
        return relations
    
    def find_similar_entities(self, entity_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """查找相似实体"""
        similarities = []
        for entity_id, embedding in self.embeddings.items():
            similarity = np.dot(entity_vector, embedding) / (
                np.linalg.norm(entity_vector) * np.linalg.norm(embedding)
            )
            similarities.append((entity_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _load_knowledge_graph(self):
        """加载知识图谱"""
        try:
            entities_file = self.storage_path / "entities.json"
            relations_file = self.storage_path / "relations.json"
            embeddings_file = self.storage_path / "embeddings.pkl"
            
            if entities_file.exists():
                with open(entities_file, 'r', encoding='utf-8') as f:
                    self.entities = json.load(f)
            
            if relations_file.exists():
                with open(relations_file, 'r', encoding='utf-8') as f:
                    self.relations = json.load(f)
                    self.relations = defaultdict(list, self.relations)
            
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
    
    def _save_knowledge_graph(self):
        """保存知识图谱"""
        try:
            entities_file = self.storage_path / "entities.json"
            relations_file = self.storage_path / "relations.json"
            embeddings_file = self.storage_path / "embeddings.pkl"
            
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(self.entities, f, ensure_ascii=False, indent=2, default=str)
            
            with open(relations_file, 'w', encoding='utf-8') as f:
                json.dump(dict(self.relations), f, ensure_ascii=False, indent=2, default=str)
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
                
        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")

class QuantumPatternRecognizer:
    """量子模式识别器"""
    
    def __init__(self):
        self.patterns = {}
        self.quantum_states = {}
        self.entanglement_matrix = None
        
    def find_similar_patterns(self, current_context: Dict[str, Any]) -> List[MemoryPattern]:
        """查找相似模式"""
        context_vector = self._encode_context(current_context)
        similar_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            similarity = self._calculate_pattern_similarity(context_vector, pattern)
            if similarity > 0.7:  # 相似度阈值
                similar_patterns.append(pattern)
        
        # 按相似度排序
        similar_patterns.sort(key=lambda p: p.confidence, reverse=True)
        return similar_patterns[:5]  # 返回前5个最相似的模式
    
    def learn_pattern(self, event: ConsciousnessEvent):
        """学习新模式"""
        pattern_signature = self._extract_pattern_signature(event)
        pattern_id = hashlib.md5(str(pattern_signature).encode()).hexdigest()
        
        if pattern_id in self.patterns:
            # 更新现有模式
            pattern = self.patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_seen = event.timestamp
            pattern.confidence = min(1.0, pattern.confidence + 0.1)
        else:
            # 创建新模式
            self.patterns[pattern_id] = MemoryPattern(
                pattern_id=pattern_id,
                pattern_type=event.event_type,
                frequency=1,
                success_rate=1.0 if event.outcome else 0.0,
                context_signature=pattern_signature,
                outcome_prediction=event.outcome,
                last_seen=event.timestamp,
                confidence=0.5
            )
    
    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """编码上下文为向量"""
        # 简化实现：将上下文转换为固定长度向量
        context_str = json.dumps(context, sort_keys=True)
        hash_obj = hashlib.sha256(context_str.encode())
        hash_hex = hash_obj.hexdigest()
        
        # 将哈希值转换为数值向量
        vector = np.array([int(hash_hex[i:i+2], 16) for i in range(0, min(len(hash_hex), 128), 2)])
        return vector / np.linalg.norm(vector)
    
    def _extract_pattern_signature(self, event: ConsciousnessEvent) -> np.ndarray:
        """提取模式签名"""
        return event.semantic_vector
    
    def _calculate_pattern_similarity(self, context_vector: np.ndarray, pattern: MemoryPattern) -> float:
        """计算模式相似度"""
        return np.dot(context_vector, pattern.context_signature) / (
            np.linalg.norm(context_vector) * np.linalg.norm(pattern.context_signature)
        )

class PredictiveEngine:
    """预测引擎"""
    
    def __init__(self):
        self.prediction_models = {}
        self.accuracy_history = defaultdict(list)
        
    def predict(self, patterns: List[MemoryPattern], current_context: Dict[str, Any]) -> Dict[str, Any]:
        """基于模式预测"""
        if not patterns:
            return {'prediction': None, 'confidence': 0.0}
        
        # 加权平均预测
        total_weight = 0
        weighted_prediction = None
        
        for pattern in patterns:
            weight = pattern.confidence * pattern.frequency
            total_weight += weight
            
            if weighted_prediction is None:
                weighted_prediction = pattern.outcome_prediction * weight
            else:
                weighted_prediction += pattern.outcome_prediction * weight
        
        if total_weight > 0:
            weighted_prediction /= total_weight
            confidence = min(1.0, total_weight / len(patterns))
        else:
            weighted_prediction = None
            confidence = 0.0
        
        return {
            'prediction': weighted_prediction,
            'confidence': confidence,
            'based_on_patterns': len(patterns)
        }
    
    def update_accuracy(self, prediction_id: str, actual_outcome: Any, predicted_outcome: Any):
        """更新预测准确率"""
        accuracy = self._calculate_accuracy(actual_outcome, predicted_outcome)
        self.accuracy_history[prediction_id].append(accuracy)
    
    def _calculate_accuracy(self, actual: Any, predicted: Any) -> float:
        """计算预测准确率"""
        if actual == predicted:
            return 1.0
        elif isinstance(actual, (int, float)) and isinstance(predicted, (int, float)):
            # 数值预测：使用相对误差
            if actual != 0:
                return 1.0 - min(1.0, abs(actual - predicted) / abs(actual))
            else:
                return 1.0 if predicted == 0 else 0.0
        else:
            # 分类预测：精确匹配
            return 0.0

class ConsciousnessStream:
    """全局意识流系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 核心组件
        self.stream_buffer = deque(maxlen=self.config.get('buffer_size', 10000))
        self.ltm_knowledge = KnowledgeGraph(self.config.get('knowledge_path'))
        self.pattern_recognizer = QuantumPatternRecognizer()
        self.predictive_engine = PredictiveEngine()
        
        # 状态管理
        self.current_context = {}
        self.active_agents = set()
        self.global_state = {}
        
        # 持久化
        self.persistence_enabled = self.config.get('persistence', True)
        self.storage_path = Path(self.config.get('storage_path', '.iflow/consciousness'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 加载历史数据
        self._load_consciousness_state()
        
        logger.info("意识流系统初始化完成")
    
    def record_event(self, event_type: str, agent_id: str, context: Dict[str, Any], 
                    outcome: Any, importance: float = 1.0) -> str:
        """记录事件到意识流"""
        with self.lock:
            # 生成事件ID
            event_id = hashlib.md5(
                f"{event_type}_{agent_id}_{time.time()}_{json.dumps(context, sort_keys=True)}".encode()
            ).hexdigest()
            
            # 编码语义向量
            semantic_vector = self._encode_semantic(context)
            
            # 计算情感权重
            emotional_weight = self._calculate_emotional_weight(context, outcome)
            
            # 创建事件
            event = ConsciousnessEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=event_type,
                agent_id=agent_id,
                context=context.copy(),
                outcome=outcome,
                semantic_vector=semantic_vector,
                emotional_weight=emotional_weight,
                importance_score=importance
            )
            
            # 添加到缓冲区
            self.stream_buffer.append(event)
            
            # 更新知识图谱
            self._update_knowledge_graph(event)
            
            # 学习模式
            self.pattern_recognizer.learn_pattern(event)
            
            # 更新全局状态
            self._update_global_state(event)
            
            # 持久化
            if self.persistence_enabled:
                self._save_event(event)
            
            logger.debug(f"记录事件: {event_type} by {agent_id}")
            return event_id
    
    def predict_next_optimal_action(self, current_context: Dict[str, Any], 
                                  agent_id: Optional[str] = None) -> Dict[str, Any]:
        """预测下一个最优行动"""
        with self.lock:
            # 更新当前上下文
            self.current_context = current_context.copy()
            
            # 查找相似模式
            similar_patterns = self.pattern_recognizer.find_similar_patterns(current_context)
            
            # 生成预测
            prediction = self.predictive_engine.predict(similar_patterns, current_context)
            
            # 添加上下文信息
            prediction['current_context'] = current_context
            prediction['agent_id'] = agent_id
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['similar_patterns_count'] = len(similar_patterns)
            
            # 如果有足够相似模式，提供详细建议
            if prediction['confidence'] > 0.7:
                prediction['recommendations'] = self._generate_recommendations(similar_patterns, current_context)
            
            return prediction
    
    def get_relevant_memories(self, query_context: Dict[str, Any], 
                            limit: int = 10) -> List[ConsciousnessEvent]:
        """获取相关记忆"""
        with self.lock:
            query_vector = self._encode_semantic(query_context)
            relevant_events = []
            
            # 从缓冲区中查找相似事件
            for event in self.stream_buffer:
                similarity = np.dot(query_vector, event.semantic_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(event.semantic_vector)
                )
                
                if similarity > 0.5:  # 相似度阈值
                    event_copy = ConsciousnessEvent(**event.__dict__)
                    event_copy.metadata['similarity'] = similarity
                    relevant_events.append(event_copy)
            
            # 按相似度排序
            relevant_events.sort(key=lambda e: e.metadata['similarity'], reverse=True)
            
            return relevant_events[:limit]
    
    def compress_and_archive(self, archive_threshold: int = 1000):
        """压缩和归档旧事件"""
        with self.lock:
            if len(self.stream_buffer) < archive_threshold:
                return
            
            # 获取最旧的事件
            old_events = list(self.stream_buffer)[:archive_threshold // 2]
            
            # 提取关键模式
            key_patterns = self._extract_key_patterns(old_events)
            
            # 创建归档记录
            archive_record = {
                'archive_id': hashlib.md5(str(time.time()).encode()).hexdigest(),
                'timestamp': datetime.now().isoformat(),
                'events_count': len(old_events),
                'key_patterns': key_patterns,
                'compressed_data': self._compress_events(old_events)
            }
            
            # 保存归档
            self._save_archive(archive_record)
            
            # 从缓冲区移除已归档事件
            for _ in range(len(old_events)):
                self.stream_buffer.popleft()
            
            logger.info(f"归档了 {len(old_events)} 个事件")
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """获取意识流摘要"""
        with self.lock:
            # 统计信息
            total_events = len(self.stream_buffer)
            event_types = defaultdict(int)
            agent_activity = defaultdict(int)
            
            for event in self.stream_buffer:
                event_types[event.event_type] += 1
                agent_activity[event.agent_id] += 1
            
            # 模式统计
            pattern_count = len(self.pattern_recognizer.patterns)
            
            # 知识图谱统计
            entity_count = len(self.ltm_knowledge.entities)
            relation_count = sum(len(relations) for relations in self.ltm_knowledge.relations.values())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_events': total_events,
                'event_types': dict(event_types),
                'agent_activity': dict(agent_activity),
                'pattern_count': pattern_count,
                'entity_count': entity_count,
                'relation_count': relation_count,
                'active_agents': list(self.active_agents),
                'global_state': self.global_state.copy()
            }
    
    def _encode_semantic(self, context: Dict[str, Any]) -> np.ndarray:
        """编码语义向量"""
        # 将上下文转换为字符串
        context_str = json.dumps(context, sort_keys=True, default=str)
        
        # 生成哈希
        hash_obj = hashlib.sha256(context_str.encode())
        hash_hex = hash_obj.hexdigest()
        
        # 转换为数值向量
        vector = np.array([int(hash_hex[i:i+2], 16) for i in range(0, min(len(hash_hex), 256), 2)])
        
        # 归一化
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector
    
    def _calculate_emotional_weight(self, context: Dict[str, Any], outcome: Any) -> float:
        """计算情感权重"""
        base_weight = 0.5
        
        # 基于结果调整权重
        if outcome is True:
            base_weight += 0.3
        elif outcome is False:
            base_weight -= 0.2
        
        # 基于上下文中的情感词调整
        emotional_keywords = ['success', 'failure', 'error', 'excellent', 'poor', 'great', 'terrible']
        context_str = str(context).lower()
        
        for keyword in emotional_keywords:
            if keyword in context_str:
                if keyword in ['success', 'excellent', 'great']:
                    base_weight += 0.1
                else:
                    base_weight -= 0.1
        
        return max(0.0, min(1.0, base_weight))
    
    def _update_knowledge_graph(self, event: ConsciousnessEvent):
        """更新知识图谱"""
        # 添加事件实体
        self.ltm_knowledge.add_entity(
            entity_id=event.event_id,
            entity_type='consciousness_event',
            properties={
                'event_type': event.event_type,
                'agent_id': event.agent_id,
                'timestamp': event.timestamp.isoformat(),
                'importance': event.importance_score
            }
        )
        
        # 存储语义向量嵌入
        self.ltm_knowledge.embeddings[event.event_id] = event.semantic_vector
        
        # 添加关系
        if event.related_events:
            for related_event_id in event.related_events:
                self.ltm_knowledge.add_relation(
                    subject=event.event_id,
                    predicate='related_to',
                    obj=related_event_id,
                    confidence=event.emotional_weight
                )
    
    def _update_global_state(self, event: ConsciousnessEvent):
        """更新全局状态"""
        # 更新活跃智能体
        self.active_agents.add(event.agent_id)
        
        # 更新全局统计
        if 'total_events' not in self.global_state:
            self.global_state['total_events'] = 0
        self.global_state['total_events'] += 1
        
        # 更新事件类型统计
        if 'event_types' not in self.global_state:
            self.global_state['event_types'] = defaultdict(int)
        self.global_state['event_types'][event.event_type] += 1
    
    def _generate_recommendations(self, patterns: List[MemoryPattern], 
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成建议"""
        recommendations = []
        
        for pattern in patterns[:3]:  # 取前3个最相关的模式
            recommendation = {
                'pattern_id': pattern.pattern_id,
                'confidence': pattern.confidence,
                'success_rate': pattern.success_rate,
                'suggestion': f"基于模式 {pattern.pattern_type}，建议采用相似策略",
                'expected_outcome': pattern.outcome_prediction
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _extract_key_patterns(self, events: List[ConsciousnessEvent]) -> List[Dict[str, Any]]:
        """提取关键模式"""
        # 简化实现：提取高频事件类型
        event_type_counts = defaultdict(int)
        for event in events:
            event_type_counts[event.event_type] += 1
        
        key_patterns = []
        for event_type, count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            key_patterns.append({
                'pattern_type': event_type,
                'frequency': count,
                'significance': count / len(events)
            })
        
        return key_patterns
    
    def _compress_events(self, events: List[ConsciousnessEvent]) -> bytes:
        """压缩事件数据"""
        # 序列化事件
        event_data = []
        for event in events:
            event_dict = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'agent_id': event.agent_id,
                'context': event.context,
                'outcome': event.outcome,
                'importance': event.importance_score
            }
            event_data.append(event_dict)
        
        # 压缩
        import gzip
        serialized = json.dumps(event_data, default=str).encode('utf-8')
        compressed = gzip.compress(serialized)
        
        return compressed
    
    def _save_event(self, event: ConsciousnessEvent):
        """保存事件"""
        event_file = self.storage_path / f"events" / f"{event.timestamp.strftime('%Y%m%d')}.jsonl"
        event_file.parent.mkdir(parents=True, exist_ok=True)
        
        event_dict = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'agent_id': event.agent_id,
            'context': event.context,
            'outcome': event.outcome,
            'semantic_vector': event.semantic_vector.tolist(),
            'emotional_weight': event.emotional_weight,
            'importance_score': event.importance_score,
            'related_events': event.related_events,
            'metadata': event.metadata
        }
        
        with open(event_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_dict, default=str) + '\n')
    
    def _save_archive(self, archive_record: Dict[str, Any]):
        """保存归档记录"""
        archive_file = self.storage_path / "archives" / f"{archive_record['archive_id']}.json"
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(archive_record, f, indent=2, default=str)
    
    def _load_consciousness_state(self):
        """加载意识流状态"""
        try:
            state_file = self.storage_path / "consciousness_state.json"
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.global_state = state.get('global_state', {})
                    self.active_agents = set(state.get('active_agents', []))
        except Exception as e:
            logger.error(f"加载意识流状态失败: {e}")
    
    def save_consciousness_state(self):
        """保存意识流状态"""
        if self.persistence_enabled:
            try:
                state_file = self.storage_path / "consciousness_state.json"
                state = {
                    'timestamp': datetime.now().isoformat(),
                    'global_state': self.global_state,
                    'active_agents': list(self.active_agents)
                }
                
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, default=str)
                    
            except Exception as e:
                logger.error(f"保存意识流状态失败: {e}")

# 全局意识流实例
_consciousness_stream_instance = None

def get_consciousness_stream(config: Dict[str, Any] = None) -> ConsciousnessStream:
    """获取全局意识流实例"""
    global _consciousness_stream_instance
    if _consciousness_stream_instance is None:
        _consciousness_stream_instance = ConsciousnessStream(config)
    return _consciousness_stream_instance

if __name__ == "__main__":
    # 测试代码
    async def test_consciousness_stream():
        # 创建意识流系统
        config = {
            'buffer_size': 1000,
            'persistence': True,
            'storage_path': '.iflow/test_consciousness'
        }
        
        consciousness = get_consciousness_stream(config)
        
        # 记录测试事件
        for i in range(10):
            event_id = consciousness.record_event(
                event_type="test_task",
                agent_id="test_agent",
                context={"task_id": i, "difficulty": i % 3},
                outcome=i % 2 == 0,
                importance=0.8
            )
            print(f"记录事件: {event_id}")
        
        # 测试预测
        prediction = consciousness.predict_next_optimal_action(
            current_context={"task_id": 11, "difficulty": 2},
            agent_id="test_agent"
        )
        
        print("\n预测结果:")
        print(json.dumps(prediction, indent=2, default=str))
        
        # 获取意识流摘要
        summary = consciousness.get_consciousness_summary()
        print("\n意识流摘要:")
        print(json.dumps(summary, indent=2, default=str))
        
        # 保存状态
        consciousness.save_consciousness_state()
    
    # 运行测试
    asyncio.run(test_consciousness_stream())