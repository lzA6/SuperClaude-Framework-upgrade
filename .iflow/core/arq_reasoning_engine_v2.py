#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧩 ARQ V2.0 专注推理与合规内核 (Enhanced Reasoning & Compliance Kernel V2.0)
Attentive Reasoning Queries & Compliance Kernel V2.0

实现增强的结构化推理系统，解决LLM长对话中的"遗忘"和"规则偏离"问题。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import enum

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class ComplianceLevel(enum.Enum):
    """合规级别"""
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"

class ReasoningMode(enum.Enum):
    """推理模式"""
    STRUCTURED = "structured"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CRITICAL = "critical"

@dataclass
class ComplianceRule:
    """合规规则"""
    rule_id: str
    rule_name: str
    rule_type: str
    description: str
    priority: int
    conditions: List[str]
    actions: List[str]
    exceptions: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    step_type: str
    content: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningChain:
    """推理链"""
    chain_id: str
    problem_statement: str
    reasoning_mode: ReasoningMode
    compliance_level: ComplianceLevel
    steps: List[ReasoningStep]
    final_conclusion: str
    confidence_score: float
    compliance_score: float
    validation_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class QuantumComplianceRules:
    """量子合规规则系统"""
    
    def __init__(self, rules_path: Optional[str] = None):
        self.rules_path = rules_path or Path.cwd() / ".iflow" / "rules"
        self.rules_path.mkdir(parents=True, exist_ok=True)
        
        self.rules = {}
        self.rule_categories = defaultdict(list)
        self.compliance_matrix = {}
        
        self._load_default_rules()
        self._load_custom_rules()
    
    def add_rule(self, rule: ComplianceRule):
        """添加规则"""
        self.rules[rule.rule_id] = rule
        self.rule_categories[rule.rule_type].append(rule.rule_id)
        self._update_compliance_matrix()
        self._save_rules()
    
    def check_compliance(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """检查合规性"""
        context = context or {}
        violations = []
        warnings = []
        suggestions = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # 检查规则条件
            if self._evaluate_conditions(rule.conditions, content, context):
                # 检查例外情况
                if not self._evaluate_exceptions(rule.exceptions, content, context):
                    violation = {
                        'rule_id': rule_id,
                        'rule_name': rule.rule_name,
                        'description': rule.description,
                        'priority': rule.priority,
                        'suggested_actions': rule.actions
                    }
                    
                    if rule.priority >= 8:
                        violations.append(violation)
                    elif rule.priority >= 5:
                        warnings.append(violation)
                    else:
                        suggestions.append(violation)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'suggestions': suggestions,
            'compliance_score': self._calculate_compliance_score(len(violations), len(warnings))
        }
    
    def get_relevant_rules(self, context: Dict[str, Any]) -> List[ComplianceRule]:
        """获取相关规则"""
        relevant_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 基于上下文匹配规则
            if self._is_rule_relevant(rule, context):
                relevant_rules.append(rule)
        
        # 按优先级排序
        relevant_rules.sort(key=lambda r: r.priority, reverse=True)
        return relevant_rules
    
    def _load_default_rules(self):
        """加载默认规则"""
        default_rules = [
            ComplianceRule(
                rule_id="no_harmful_content",
                rule_name="无有害内容",
                rule_type="safety",
                description="不得生成任何有害、危险或非法的内容",
                priority=10,
                conditions=["harmful_keywords", "illegal_activities"],
                actions=["refuse_request", "suggest_alternatives"]
            ),
            ComplianceRule(
                rule_id="accuracy_required",
                rule_name="准确性要求",
                rule_type="quality",
                description="所有信息必须准确可靠，不得传播虚假信息",
                priority=9,
                conditions=["factual_claims", "statistics", "technical_specifications"],
                actions=["verify_facts", "cite_sources", "express_uncertainty"]
            ),
            ComplianceRule(
                rule_id="privacy_protection",
                rule_name="隐私保护",
                rule_type="privacy",
                description="保护用户隐私，不得泄露个人信息",
                priority=10,
                conditions=["personal_data", "identifyingInformation"],
                actions=["anonymize_data", "refuse_share", "explain_limits"]
            ),
            ComplianceRule(
                rule_id="ethical_considerations",
                rule_name="伦理考虑",
                rule_type="ethics",
                description="考虑伦理影响，避免偏见和歧视",
                priority=8,
                conditions=["demographic_groups", "sensitive_topics", "biases"],
                actions=["ensure_fairness", "provide_balance", "acknowledge_complexity"]
            ),
            ComplianceRule(
                rule_id="code_quality",
                rule_name="代码质量",
                rule_type="technical",
                description="生成高质量、安全、可维护的代码",
                priority=7,
                conditions=["code_generation", "security_practices", "performance"],
                actions=["follow_best_practices", "add_comments", "include_error_handling"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
            self.rule_categories[rule.rule_type].append(rule.rule_id)
    
    def _load_custom_rules(self):
        """加载自定义规则"""
        try:
            rules_file = self.rules_path / "custom_rules.json"
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf-8') as f:
                    custom_rules_data = json.load(f)
                    
                    for rule_data in custom_rules_data.get('rules', []):
                        rule = ComplianceRule(**rule_data)
                        self.rules[rule.rule_id] = rule
                        self.rule_categories[rule.rule_type].append(rule.rule_id)
                        
        except Exception as e:
            logger.error(f"加载自定义规则失败: {e}")
    
    def _save_rules(self):
        """保存规则"""
        try:
            rules_file = self.rules_path / "custom_rules.json"
            custom_rules = []
            
            for rule in self.rules.values():
                rule_dict = {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'rule_type': rule.rule_type,
                    'description': rule.description,
                    'priority': rule.priority,
                    'conditions': rule.conditions,
                    'actions': rule.actions,
                    'exceptions': rule.exceptions,
                    'enabled': rule.enabled,
                    'metadata': rule.metadata
                }
                custom_rules.append(rule_dict)
            
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump({'rules': custom_rules}, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存规则失败: {e}")
    
    def _evaluate_conditions(self, conditions: List[str], content: str, context: Dict[str, Any]) -> bool:
        """评估条件"""
        for condition in conditions:
            if self._evaluate_condition(condition, content, context):
                return True
        return False
    
    def _evaluate_condition(self, condition: str, content: str, context: Dict[str, Any]) -> bool:
        """评估单个条件"""
        content_lower = content.lower()
        
        # 关键词匹配
        if condition == "harmful_keywords":
            harmful_keywords = ['hack', 'exploit', 'malware', 'virus', 'attack', 'weapon']
            return any(keyword in content_lower for keyword in harmful_keywords)
        
        elif condition == "illegal_activities":
            illegal_keywords = ['illegal', 'crime', 'fraud', 'theft', 'drugs', 'violence']
            return any(keyword in content_lower for keyword in illegal_keywords)
        
        elif condition == "factual_claims":
            # 检测事实性声明
            patterns = [r'\b(is|are|was|were)\b', r'\d+%?', r'\$[\d,]+']
            return any(re.search(pattern, content) for pattern in patterns)
        
        elif condition == "personal_data":
            personal_patterns = [r'\b\d{3}-\d{2}-\d{4}\b', r'\b\d{11}\b', r'\w+@\w+\.\w+']
            return any(re.search(pattern, content) for pattern in personal_patterns)
        
        # 其他条件评估...
        
        return False
    
    def _evaluate_exceptions(self, exceptions: List[str], content: str, context: Dict[str, Any]) -> bool:
        """评估例外情况"""
        for exception in exceptions:
            if self._evaluate_condition(exception, content, context):
                return True
        return False
    
    def _is_rule_relevant(self, rule: ComplianceRule, context: Dict[str, Any]) -> bool:
        """判断规则是否相关"""
        # 基于上下文判断规则相关性
        if 'task_type' in context:
            if rule.rule_type == 'safety' and context['task_type'] in ['coding', 'writing']:
                return True
            elif rule.rule_type == 'technical' and context['task_type'] == 'coding':
                return True
            elif rule.rule_type == 'privacy' and 'personal_data' in context:
                return True
        
        return True  # 默认相关
    
    def _calculate_compliance_score(self, violations: int, warnings: int) -> float:
        """计算合规分数"""
        base_score = 100.0
        violation_penalty = violations * 20
        warning_penalty = warnings * 5
        
        return max(0.0, base_score - violation_penalty - warning_penalty)
    
    def _update_compliance_matrix(self):
        """更新合规矩阵"""
        # 更新规则间的依赖关系和冲突检测
        self.compliance_matrix = {}
        
        for rule_id, rule in self.rules.items():
            self.compliance_matrix[rule_id] = {
                'dependencies': [],
                'conflicts': []
            }

class AdvancedReasoningTemplates:
    """高级推理模板"""
    
    def __init__(self):
        self.templates = {}
        self._load_templates()
    
    def get_template(self, reasoning_mode: ReasoningMode, problem_type: str) -> Dict[str, Any]:
        """获取推理模板"""
        template_key = f"{reasoning_mode.value}_{problem_type}"
        return self.templates.get(template_key, self._get_default_template())
    
    def _load_templates(self):
        """加载推理模板"""
        # 结构化推理模板
        self.templates["structured_analytical"] = {
            'steps': [
                'problem_identification',
                'information_gathering',
                'hypothesis_formation',
                'evidence_evaluation',
                'logical_deduction',
                'conclusion_validation'
            ],
            'prompts': {
                'problem_identification': "明确问题的核心目标和约束条件",
                'information_gathering': "收集所有相关信息和数据",
                'hypothesis_formation': "基于信息形成初步假设",
                'evidence_evaluation': "评估证据支持假设的程度",
                'logical_deduction': "进行逻辑推理得出结论",
                'conclusion_validation': "验证结论的合理性和可靠性"
            }
        }
        
        # 创新推理模板
        self.templates["creative_design"] = {
            'steps': [
                'requirement_analysis',
                'ideation_brainstorm',
                'concept_development',
                'feasibility_assessment',
                'prototype_design',
                'iteration_refinement'
            ],
            'prompts': {
                'requirement_analysis': "深入分析用户需求和约束条件",
                'ideation_brainstorm': "进行创造性思维和头脑风暴",
                'concept_development': "发展具体的概念和方案",
                'feasibility_assessment': "评估技术可行性和资源需求",
                'prototype_design': "设计原型和验证方案",
                'iteration_refinement': "基于反馈迭代改进方案"
            }
        }
        
        # 批判性思维模板
        self.templates["critical_evaluation"] = {
            'steps': [
                'claim_identification',
                'evidence_examination',
                'bias_detection',
                'logical_analysis',
                'alternative_consideration',
                'judgment_formation'
            ],
            'prompts': {
                'claim_identification': "识别需要评估的核心主张",
                'evidence_examination': "仔细检查支持证据的质量和相关性",
                'bias_detection': "检测潜在的偏见和假设",
                'logical_analysis': "分析论证的逻辑结构",
                'alternative_consideration': "考虑替代观点和解释",
                'judgment_formation': "基于分析形成判断"
            }
        }
    
    def _get_default_template(self) -> Dict[str, Any]:
        """获取默认模板"""
        return {
            'steps': ['understand', 'analyze', 'solve', 'verify'],
            'prompts': {
                'understand': "理解问题的本质",
                'analyze': "分析问题的关键要素",
                'solve': "寻找解决方案",
                'verify': "验证解决方案的有效性"
            }
        }

class FormalValidationEngine:
    """形式化验证引擎"""
    
    def __init__(self):
        self.validation_rules = {}
        self.logic_checkers = {}
        self._initialize_validators()
    
    def validate_reasoning(self, reasoning_chain: ReasoningChain) -> Dict[str, Any]:
        """验证推理链"""
        validation_results = {
            'logical_consistency': self._check_logical_consistency(reasoning_chain),
            'evidence_coherence': self._check_evidence_coherence(reasoning_chain),
            'conclusion_validity': self._check_conclusion_validity(reasoning_chain),
            'completeness': self._check_completeness(reasoning_chain),
            'overall_score': 0.0
        }
        
        # 计算总分
        scores = [
            validation_results['logical_consistency']['score'],
            validation_results['evidence_coherence']['score'],
            validation_results['conclusion_validity']['score'],
            validation_results['completeness']['score']
        ]
        validation_results['overall_score'] = sum(scores) / len(scores)
        
        return validation_results
    
    def _check_logical_consistency(self, reasoning_chain: ReasoningChain) -> Dict[str, Any]:
        """检查逻辑一致性"""
        contradictions = []
        consistency_score = 1.0
        
        # 检查步骤间的逻辑关系
        for i, step in enumerate(reasoning_chain.steps[:-1]):
            next_step = reasoning_chain.steps[i + 1]
            
            # 检查结论是否与下一步的假设一致
            if step.conclusions:
                for conclusion in step.conclusions:
                    if conclusion in next_step.assumptions:
                        # 一致，继续
                        continue
                    elif self._is_contradiction(conclusion, next_step.assumptions):
                        contradictions.append(f"步骤 {i+1} 和 {i+2} 之间存在逻辑矛盾")
                        consistency_score -= 0.2
        
        return {
            'score': max(0.0, consistency_score),
            'contradictions': contradictions,
            'status': 'consistent' if not contradictions else 'inconsistent'
        }
    
    def _check_evidence_coherence(self, reasoning_chain: ReasoningChain) -> Dict[str, Any]:
        """检查证据连贯性"""
        evidence_score = 1.0
        missing_evidence = []
        
        for step in reasoning_chain.steps:
            if step.assumptions and not step.evidence:
                missing_evidence.append(f"步骤 {step.step_id} 缺乏支持证据")
                evidence_score -= 0.1
        
        return {
            'score': max(0.0, evidence_score),
            'missing_evidence': missing_evidence,
            'status': 'coherent' if not missing_evidence else 'needs_evidence'
        }
    
    def _check_conclusion_validity(self, reasoning_chain: ReasoningChain) -> Dict[str, Any]:
        """检查结论有效性"""
        if not reasoning_chain.steps:
            return {'score': 0.0, 'status': 'no_steps'}
        
        last_step = reasoning_chain.steps[-1]
        conclusion_score = 1.0
        
        # 检查结论是否得到前面步骤的支持
        if last_step.conclusions:
            for conclusion in last_step.conclusions:
                if not self._is_supported_by_evidence(conclusion, reasoning_chain.steps[:-1]):
                    conclusion_score -= 0.2
        
        return {
            'score': max(0.0, conclusion_score),
            'status': 'valid' if conclusion_score > 0.8 else 'questionable'
        }
    
    def _check_completeness(self, reasoning_chain: ReasoningChain) -> Dict[str, Any]:
        """检查完整性"""
        completeness_score = 1.0
        missing_elements = []
        
        # 检查是否有明确的问题定义
        if not reasoning_chain.problem_statement:
            missing_elements.append("缺少明确的问题定义")
            completeness_score -= 0.3
        
        # 检查推理步骤是否充分
        if len(reasoning_chain.steps) < 3:
            missing_elements.append("推理步骤不充分")
            completeness_score -= 0.2
        
        # 检查是否有最终结论
        if not reasoning_chain.final_conclusion:
            missing_elements.append("缺少最终结论")
            completeness_score -= 0.3
        
        return {
            'score': max(0.0, completeness_score),
            'missing_elements': missing_elements,
            'status': 'complete' if completeness_score > 0.8 else 'incomplete'
        }
    
    def _is_contradiction(self, statement: str, assumptions: List[str]) -> bool:
        """检查是否存在矛盾"""
        # 简化的矛盾检测
        contradictory_pairs = [
            ('true', 'false'),
            ('yes', 'no'),
            ('enable', 'disable'),
            ('increase', 'decrease')
        ]
        
        for pair in contradictory_pairs:
            if pair[0] in statement.lower() and any(pair[1] in assumption.lower() for assumption in assumptions):
                return True
            if pair[1] in statement.lower() and any(pair[0] in assumption.lower() for assumption in assumptions):
                return True
        
        return False
    
    def _is_supported_by_evidence(self, conclusion: str, steps: List[ReasoningStep]) -> bool:
        """检查结论是否得到证据支持"""
        conclusion_keywords = conclusion.lower().split()
        
        for step in steps:
            for evidence in step.evidence:
                evidence_keywords = evidence.lower().split()
                # 简化的证据支持检查
                if any(keyword in evidence_keywords for keyword in conclusion_keywords):
                    return True
        
        return False
    
    def _initialize_validators(self):
        """初始化验证器"""
        self.validation_rules = {
            'logical_consistency': self._check_logical_consistency,
            'evidence_coherence': self._check_evidence_coherence,
            'conclusion_validity': self._check_conclusion_validity,
            'completeness': self._check_completeness
        }

class ARQReasoningEngineV2:
    """ARQ推理引擎V2.0"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 核心组件
        self.compliance_rules = QuantumComplianceRules(self.config.get('rules_path'))
        self.reasoning_templates = AdvancedReasoningTemplates()
        self.validation_engine = FormalValidationEngine()
        
        # 推理历史
        self.reasoning_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # 配置
        self.default_compliance_level = ComplianceLevel(self.config.get('compliance_level', 'strict'))
        self.default_reasoning_mode = ReasoningMode(self.config.get('reasoning_mode', 'structured'))
        
        logger.info("ARQ推理引擎V2.0初始化完成")
    
    def structured_reasoning(self, problem_statement: str, context: Dict[str, Any] = None,
                           reasoning_mode: Optional[ReasoningMode] = None,
                           compliance_level: Optional[ComplianceLevel] = None) -> ReasoningChain:
        """结构化推理过程"""
        context = context or {}
        reasoning_mode = reasoning_mode or self.default_reasoning_mode
        compliance_level = compliance_level or self.default_compliance_level
        
        # 生成推理链ID
        chain_id = hashlib.md5(f"{problem_statement}_{time.time()}".encode()).hexdigest()
        
        # 获取推理模板
        problem_type = self._classify_problem_type(problem_statement, context)
        template = self.reasoning_templates.get_template(reasoning_mode, problem_type)
        
        # 执行推理步骤
        steps = []
        current_context = context.copy()
        
        for step_name in template['steps']:
            step_prompt = template['prompts'].get(step_name, f"执行{step_name}")
            step = self._execute_reasoning_step(
                step_name, step_prompt, problem_statement, current_context
            )
            steps.append(step)
            
            # 更新上下文
            current_context.update({
                'last_step': step_name,
                'last_conclusions': step.conclusions,
                'accumulated_evidence': current_context.get('accumulated_evidence', []) + step.evidence
            })
        
        # 生成最终结论
        final_conclusion = self._generate_final_conclusion(steps)
        
        # 创建推理链
        reasoning_chain = ReasoningChain(
            chain_id=chain_id,
            problem_statement=problem_statement,
            reasoning_mode=reasoning_mode,
            compliance_level=compliance_level,
            steps=steps,
            final_conclusion=final_conclusion,
            confidence_score=self._calculate_confidence_score(steps),
            compliance_score=self._calculate_compliance_score(steps, context)
        )
        
        # 验证推理链
        validation_results = self.validation_engine.validate_reasoning(reasoning_chain)
        reasoning_chain.validation_results = validation_results
        
        # 记录推理历史
        self.reasoning_history.append(reasoning_chain)
        
        # 更新性能指标
        self._update_performance_metrics(reasoning_chain)
        
        return reasoning_chain
    
    def check_compliance(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """检查内容合规性"""
        return self.compliance_rules.check_compliance(content, context)
    
    def get_reasoning_insights(self, chain_id: str) -> Dict[str, Any]:
        """获取推理洞察"""
        for chain in self.reasoning_history:
            if chain.chain_id == chain_id:
                return {
                    'chain_id': chain.chain_id,
                    'reasoning_mode': chain.reasoning_mode.value,
                    'compliance_level': chain.compliance_level.value,
                    'step_count': len(chain.steps),
                    'confidence_score': chain.confidence_score,
                    'compliance_score': chain.compliance_score,
                    'validation_summary': chain.validation_results,
                    'key_insights': self._extract_key_insights(chain)
                }
        
        return {'error': f'未找到推理链: {chain_id}'}
    
    def optimize_reasoning_strategy(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """优化推理策略"""
        if not performance_history:
            return {'recommendations': ['收集更多性能数据']}
        
        # 分析性能趋势
        avg_confidence = sum(p.get('confidence_score', 0) for p in performance_history) / len(performance_history)
        avg_compliance = sum(p.get('compliance_score', 0) for p in performance_history) / len(performance_history)
        
        recommendations = []
        
        if avg_confidence < 0.8:
            recommendations.append("建议增加证据收集和分析步骤")
        
        if avg_compliance < 0.9:
            recommendations.append("建议加强合规性检查")
        
        # 分析最佳实践
        best_performances = sorted(performance_history, key=lambda x: x.get('overall_score', 0), reverse=True)[:5]
        common_patterns = self._identify_common_patterns(best_performances)
        
        return {
            'current_performance': {
                'avg_confidence': avg_confidence,
                'avg_compliance': avg_compliance
            },
            'recommendations': recommendations,
            'best_practices': common_patterns,
            'optimization_suggestions': self._generate_optimization_suggestions(common_patterns)
        }
    
    def _classify_problem_type(self, problem_statement: str, context: Dict[str, Any]) -> str:
        """分类问题类型"""
        statement_lower = problem_statement.lower()
        
        if any(keyword in statement_lower for keyword in ['design', 'create', 'build', 'develop']):
            return 'design'
        elif any(keyword in statement_lower for keyword in ['analyze', 'evaluate', 'assess', 'review']):
            return 'analytical'
        elif any(keyword in statement_lower for keyword in ['solve', 'fix', 'resolve', 'address']):
            return 'problem_solving'
        elif any(keyword in statement_lower for keyword in ['decide', 'choose', 'select', 'recommend']):
            return 'decision'
        else:
            return 'general'
    
    def _execute_reasoning_step(self, step_name: str, step_prompt: str, 
                              problem_statement: str, context: Dict[str, Any]) -> ReasoningStep:
        """执行推理步骤"""
        step_id = hashlib.md5(f"{step_name}_{time.time()}".encode()).hexdigest()
        
        # 模拟推理步骤执行（实际应用中会调用LLM）
        step_content = f"步骤: {step_name}\n提示: {step_prompt}\n问题: {problem_statement}"
        
        # 基于步骤类型生成相应内容
        if step_name == 'information_gathering':
            evidence = ["收集相关信息", "分析数据源", "验证信息准确性"]
            assumptions = ["信息是可靠的", "数据是完整的"]
            conclusions = ["信息收集完成", "已获得足够数据"]
        elif step_name == 'hypothesis_formation':
            evidence = context.get('accumulated_evidence', [])
            assumptions = ["假设基于已有信息", "假设是可验证的"]
            conclusions = ["形成初步假设", "假设需要进一步验证"]
        else:
            evidence = []
            assumptions = []
            conclusions = [f"完成{step_name}步骤"]
        
        return ReasoningStep(
            step_id=step_id,
            step_type=step_name,
            content=step_content,
            confidence=0.8,
            evidence=evidence,
            assumptions=assumptions,
            conclusions=conclusions,
            next_steps=[]
        )
    
    def _generate_final_conclusion(self, steps: List[ReasoningStep]) -> str:
        """生成最终结论"""
        if not steps:
            return "无法生成结论：缺少推理步骤"
        
        last_step = steps[-1]
        if last_step.conclusions:
            return " ".join(last_step.conclusions)
        
        return "基于推理分析得出结论"
    
    def _calculate_confidence_score(self, steps: List[ReasoningStep]) -> float:
        """计算置信度分数"""
        if not steps:
            return 0.0
        
        total_confidence = sum(step.confidence for step in steps)
        return total_confidence / len(steps)
    
    def _calculate_compliance_score(self, steps: List[ReasoningStep], context: Dict[str, Any]) -> float:
        """计算合规分数"""
        # 检查所有步骤的合规性
        total_score = 100.0
        
        for step in steps:
            compliance_result = self.compliance_rules.check_compliance(step.content, context)
            if not compliance_result['compliant']:
                total_score -= len(compliance_result['violations']) * 10
                total_score -= len(compliance_result['warnings']) * 5
        
        return max(0.0, total_score)
    
    def _extract_key_insights(self, reasoning_chain: ReasoningChain) -> List[str]:
        """提取关键洞察"""
        insights = []
        
        # 从推理步骤中提取洞察
        for step in reasoning_chain.steps:
            if step.conclusions:
                insights.extend(step.conclusions)
        
        # 添加验证结果洞察
        validation = reasoning_chain.validation_results
        if validation['overall_score'] > 0.9:
            insights.append("推理质量优秀")
        elif validation['overall_score'] < 0.7:
            insights.append("推理需要改进")
        
        return insights[:5]  # 返回前5个洞察
    
    def _identify_common_patterns(self, performances: List[Dict[str, Any]]) -> List[str]:
        """识别共同模式"""
        patterns = []
        
        # 分析成功案例的共同特征
        high_score_cases = [p for p in performances if p.get('overall_score', 0) > 0.8]
        
        if high_score_cases:
            # 统计推理模式
            reasoning_modes = [p.get('reasoning_mode', 'structured') for p in high_score_cases]
            most_common_mode = max(set(reasoning_modes), key=reasoning_modes.count)
            patterns.append(f"最常用的推理模式: {most_common_mode}")
            
            # 统计步骤数量
            step_counts = [p.get('step_count', 0) for p in high_score_cases]
            avg_steps = sum(step_counts) / len(step_counts)
            patterns.append(f"平均推理步骤数: {avg_steps:.1f}")
        
        return patterns
    
    def _generate_optimization_suggestions(self, patterns: List[str]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        for pattern in patterns:
            if "推理模式" in pattern:
                suggestions.append("继续使用有效的推理模式")
            elif "步骤数" in pattern:
                suggestions.append("保持适当的推理详细程度")
        
        # 通用优化建议
        suggestions.extend([
            "定期更新合规规则",
            "持续监控推理质量",
            "收集用户反馈以改进推理"
        ])
        
        return suggestions
    
    def _update_performance_metrics(self, reasoning_chain: ReasoningChain):
        """更新性能指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'chain_id': reasoning_chain.chain_id,
            'reasoning_mode': reasoning_chain.reasoning_mode.value,
            'compliance_level': reasoning_chain.compliance_level.value,
            'step_count': len(reasoning_chain.steps),
            'confidence_score': reasoning_chain.confidence_score,
            'compliance_score': reasoning_chain.compliance_score,
            'overall_score': reasoning_chain.validation_results.get('overall_score', 0.0)
        }
        
        self.performance_metrics['reasoning_chains'].append(metrics)

# 全局ARQ推理引擎实例
_arq_engine_instance = None

def get_arq_engine(config: Dict[str, Any] = None) -> ARQReasoningEngineV2:
    """获取ARQ推理引擎实例"""
    global _arq_engine_instance
    if _arq_engine_instance is None:
        _arq_engine_instance = ARQReasoningEngineV2(config)
    return _arq_engine_instance

if __name__ == "__main__":
    # 测试代码
    import time
    
    def test_arq_engine():
        # 创建ARQ引擎
        config = {
            'compliance_level': 'strict',
            'reasoning_mode': 'structured'
        }
        
        arq_engine = get_arq_engine(config)
        
        # 测试结构化推理
        problem = "如何设计一个安全高效的用户认证系统？"
        context = {
            'task_type': 'design',
            'security_requirements': ['authentication', 'authorization', 'encryption']
        }
        
        reasoning_chain = arq_engine.structured_reasoning(
            problem_statement=problem,
            context=context
        )
        
        print("推理链结果:")
        print(f"问题: {reasoning_chain.problem_statement}")
        print(f"推理模式: {reasoning_chain.reasoning_mode.value}")
        print(f"合规级别: {reasoning_chain.compliance_level.value}")
        print(f"置信度: {reasoning_chain.confidence_score:.2f}")
        print(f"合规分数: {reasoning_chain.compliance_score:.2f}")
        print(f"最终结论: {reasoning_chain.final_conclusion}")
        
        print("\n验证结果:")
        validation = reasoning_chain.validation_results
        for key, value in validation.items():
            if isinstance(value, dict) and 'score' in value:
                print(f"{key}: {value['score']:.2f} ({value['status']})")
        
        # 测试合规性检查
        test_content = "这是一个包含用户密码和信用卡号的内容"
        compliance_result = arq_engine.check_compliance(test_content)
        
        print("\n合规性检查结果:")
        print(f"合规: {compliance_result['compliant']}")
        print(f"合规分数: {compliance_result['compliance_score']}")
        
        if compliance_result['violations']:
            print("违规项:")
            for violation in compliance_result['violations']:
                print(f"  - {violation['rule_name']}: {violation['description']}")
    
    # 运行测试
    test_arq_engine()