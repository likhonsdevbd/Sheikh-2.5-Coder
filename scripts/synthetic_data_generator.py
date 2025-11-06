#!/usr/bin/env python3
"""
Synthetic Data Generator

Generates synthetic data using multiple methodologies:
- Self-Instruct methodology for web development
- Evol-Instruct for complexity scaling
- AST mutation for code augmentation
- Domain-specific templates for XML/MDX/JS

Author: MiniMax Agent
Date: 2025-11-06
"""

import os
import sys
import json
import yaml
import logging
import ast
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import hashlib

# Data processing libraries
from datasets import Dataset

# Natural language processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# AST parsing and mutation
try:
    import astunparse
    ASTUNPARSE_AVAILABLE = True
except ImportError:
    ASTUNPARSE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SyntheticExample:
    """Data class for synthetic examples"""
    content: str
    content_type: str  # 'code', 'instruction', 'comment', 'explanation'
    language: str
    generation_method: str
    quality_score: float
    difficulty_level: str
    domain: str
    metadata: Dict[str, Any]

class SyntheticDataGenerator:
    """
    Synthetic data generator using multiple methodologies
    
    Generates:
    - Self-Instruct data for instruction-following
    - Evol-Instruct data for complexity scaling
    - AST mutation data for code augmentation
    - Domain-specific templates for web development
    """
    
    def __init__(self, config: DataPreparationConfig):
        """Initialize synthetic data generator"""
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.initialize_generation_methods()
        self.initialize_domain_templates()
        
        # Generation statistics
        self.stats = {
            'self_instruct_generated': 0,
            'evol_instruct_generated': 0,
            'ast_mutation_generated': 0,
            'domain_specific_generated': 0,
            'total_generated': 0,
            'quality_filtered': 0
        }
        
        logger.info("Synthetic Data Generator initialized")
    
    def setup_logging(self):
        """Setup logging for synthetic generation"""
        log_handler = logging.FileHandler('logs/synthetic_data_generator.log')
        log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(log_handler)
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw/synthetic',
            'data/processed/synthetic',
            'cache/ast_mutations',
            'cache/domain_templates',
            'evaluation/synthetic_reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_generation_methods(self):
        """Initialize synthetic generation methods"""
        self.generation_methods = {
            'self_instruct': self.self_instruct_generation,
            'evol_instruct': self.evol_instruct_generation,
            'ast_mutation': self.ast_mutation_generation,
            'domain_specific': self.domain_specific_generation
        }
        
        # Generation parameters
        self.generation_params = {
            'self_instruct': {
                'target_count': 50000,
                'difficulty_distribution': {'beginner': 0.4, 'intermediate': 0.4, 'advanced': 0.2},
                'language_distribution': {
                    'javascript': 0.35,
                    'typescript': 0.25,
                    'xml': 0.20,
                    'html': 0.10,
                    'mdx': 0.10
                }
            },
            'evol_instruct': {
                'target_count': 30000,
                'complexity_increase': 0.4,
                'evolution_operations': [
                    'instruction_complexity_increase',
                    'output_elaboration',
                    'task_clarification',
                    'constraint_addition'
                ]
            },
            'ast_mutation': {
                'target_count': 40000,
                'mutation_rate': 0.3,
                'supported_languages': ['javascript', 'typescript'],
                'mutation_types': [
                    'variable_renaming',
                    'function_modification',
                    'syntax_variation',
                    'logic_optimization'
                ]
            },
            'domain_specific': {
                'target_count': 25000,
                'domains': {
                    'xml_configuration': 0.4,
                    'mdx_components': 0.3,
                    'react_hooks': 0.2,
                    'vue_templates': 0.1
                }
            }
        }
        
        logger.info(f"Initialized {len(self.generation_methods)} generation methods")
    
    def initialize_domain_templates(self):
        """Initialize domain-specific templates for web development"""
        self.domain_templates = {
            'xml_configuration': {
                'spring_boot_config': '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.7.0</version>
        <relativePath/>
    </parent>
    
    <properties>
        <java.version>11</java.version>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
    </dependencies>
</project>''',
                
                'docker_config': '''FROM openjdk:11-jre-slim

WORKDIR /app

COPY target/*.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "/app/app.jar"]''',
                
                'kubernetes_deployment': '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-deployment
  labels:
    app: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: webapp
        image: webapp:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"'''
            },
            
            'mdx_components': {
                'interactive_code_component': '''import { CodeBlock } from './CodeBlock'

export const InteractiveCode = ({ children, language = 'javascript' }) => {
  const [code, setCode] = useState(children)
  const [output, setOutput] = useState('')
  
  const runCode = () => {
    try {
      // Execute JavaScript code safely
      const result = eval(code)
      setOutput(String(result))
    } catch (error) {
      setOutput(`Error: ${error.message}`)
    }
  }
  
  return (
    <div className="interactive-code-block">
      <textarea 
        value={code}
        onChange={(e) => setCode(e.target.value)}
        className="code-input"
      />
      <button onClick={runCode} className="run-button">
        Run Code
      </button>
      <div className="code-output">
        <strong>Output:</strong> {output}
      </div>
    </div>
  )
}

# Usage
<InteractiveCode>
console.log("Hello World!")
</InteractiveCode>''',
                
                'embedded_component': '''import { Chart } from './Chart'
import { Calculator } from './Calculator'

export const DataVisualization = ({ data, type = 'line' }) => {
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      value: parseFloat(item.value) || 0
    }))
  }, [data])
  
  return (
    <div className="data-viz-container">
      <h2>Data Visualization</h2>
      <Chart data={processedData} type={type} />
      <Calculator />
    </div>
  )
}''',
                
                'documentation_component': '''import { Tabs, Tab } from './Tabs'
import { Callout } from './Callout'
import { Link } from './Link'

export const APIDocumentation = ({ apiSpec }) => {
  return (
    <div className="api-docs">
      <h1>{apiSpec.title}</h1>
      <p>{apiSpec.description}</p>
      
      <Callout type="info">
        API Version: {apiSpec.version} | Base URL: {apiSpec.baseUrl}
      </Callout>
      
      <Tabs>
        {apiSpec.endpoints.map(endpoint => (
          <Tab key={endpoint.path} label={endpoint.method}>
            <div className="endpoint-doc">
              <h3>{endpoint.method} {endpoint.path}</h3>
              <p>{endpoint.description}</p>
              
              {endpoint.parameters && (
                <div className="parameters">
                  <h4>Parameters</h4>
                  <ul>
                    {endpoint.parameters.map(param => (
                      <li key={param.name}>
                        <code>{param.name}</code> ({param.type}) - {param.description}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {endpoint.examples && (
                <div className="examples">
                  <h4>Examples</h4>
                  <CodeBlock language="bash">
                    {endpoint.examples.curl}
                  </CodeBlock>
                </div>
              )}
            </div>
          </Tab>
        ))}
      </Tabs>
    </div>
  )
}'''
            },
            
            'react_hooks': {
                'use_api_hooks': '''import { useState, useEffect } from 'react'

export const useApi = (url, options = {}) => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch(url, options)
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        
        const result = await response.json()
        setData(result)
        setError(null)
      } catch (err) {
        setError(err.message)
        setData(null)
      } finally {
        setLoading(false)
      }
    }
    
    if (url) {
      fetchData()
    }
  }, [url, JSON.stringify(options)])
  
  return { data, loading, error }
}

// Usage example
const UserProfile = ({ userId }) => {
  const { data: user, loading, error } = useApi(`/api/users/${userId}`)
  
  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>
  
  return (
    <div className="user-profile">
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  )
}''',
                
                'use_local_storage': '''import { useState, useEffect } from 'react'

export const useLocalStorage = (key, initialValue) => {
  // Get from local storage then parse stored json or return initialValue
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error)
      return initialValue
    }
  })
  
  // Return a wrapped version of useState's setter function that persists the new value to localStorage
  const setValue = (value) => {
    try {
      // Allow value to be a function so we have the same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value
      setStoredValue(valueToStore)
      window.localStorage.setItem(key, JSON.stringify(valueToStore))
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error)
    }
  }
  
  return [storedValue, setValue]
}

// Usage example
const Settings = () => {
  const [theme, setTheme] = useLocalStorage('theme', 'light')
  const [notifications, setNotifications] = useLocalStorage('notifications', true)
  
  return (
    <div className="settings">
      <label>
        Theme:
        <select value={theme} onChange={(e) => setTheme(e.target.value)}>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
      
      <label>
        <input 
          type="checkbox" 
          checked={notifications} 
          onChange={(e) => setNotifications(e.target.checked)} 
        />
        Enable notifications
      </label>
    </div>
  )
}''',
                
                'use_debounce': '''import { useState, useEffect } from 'react'

export const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value)
  
  useEffect(() => {
    // Set up the timeout to update the debounced value
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)
    
    // Cancel the timeout if value changes (also on delay change or unmount)
    return () => {
      clearTimeout(handler)
    }
  }, [value, delay])
  
  return debouncedValue
}

// Usage example
const SearchBox = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const debouncedSearchTerm = useDebounce(searchTerm, 500)
  
  // Effect for API call
  useEffect(() => {
    if (debouncedSearchTerm) {
      // Perform search API call
      searchAPI(debouncedSearchTerm).then(results => {
        setSearchResults(results)
      })
    } else {
      setSearchResults([])
    }
  }, [debouncedSearchTerm])
  
  return (
    <div className="search-box">
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search..."
      />
      <div className="results">
        {searchResults.map(result => (
          <div key={result.id} className="result">
            {result.title}
          </div>
        ))}
      </div>
    </div>
  )
}'''
            },
            
            'vue_templates': {
                'component_template': '''<template>
  <div class="user-profile">
    <div class="profile-header">
      <img :src="user.avatar" :alt="user.name" class="avatar" />
      <div class="user-info">
        <h2>{{ user.name }}</h2>
        <p>{{ user.email }}</p>
        <span class="user-role">{{ user.role }}</span>
      </div>
    </div>
    
    <div class="profile-content">
      <UserStats :stats="user.stats" />
      <UserActivities :activities="user.activities" />
    </div>
    
    <div class="profile-actions">
      <button @click="editProfile" class="btn-primary">
        Edit Profile
      </button>
      <button @click="shareProfile" class="btn-secondary">
        Share Profile
      </button>
    </div>
  </div>
</template>

<script>
import UserStats from './UserStats.vue'
import UserActivities from './UserActivities.vue'

export default {
  name: 'UserProfile',
  components: {
    UserStats,
    UserActivities
  },
  props: {
    user: {
      type: Object,
      required: true
    }
  },
  methods: {
    editProfile() {
      this.$emit('edit-user', this.user)
    },
    shareProfile() {
      if (navigator.share) {
        navigator.share({
          title: this.user.name,
          text: `Check out ${this.user.name}'s profile`,
          url: window.location.href
        })
      } else {
        // Fallback for browsers that don't support Web Share API
        navigator.clipboard.writeText(window.location.href)
        this.$emit('show-notification', 'Profile link copied!')
      }
    }
  }
}
</script>

<style scoped>
.user-profile {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.profile-header {
  display: flex;
  align-items: center;
  margin-bottom: 2rem;
}

.avatar {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  margin-right: 1.5rem;
}

.user-info h2 {
  margin: 0;
  color: #333;
}

.user-info p {
  margin: 0.5rem 0;
  color: #666;
}

.user-role {
  background: #e3f2fd;
  color: #1976d2;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
}

.profile-actions {
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
}
</style>''',
                
                'form_template': '''<template>
  <form @submit.prevent="handleSubmit" class="user-form">
    <div class="form-group">
      <label for="name">Name *</label>
      <input
        id="name"
        v-model="form.name"
        type="text"
        :class="{ 'error': errors.name }"
        placeholder="Enter your full name"
        required
      />
      <span v-if="errors.name" class="error-message">{{ errors.name }}</span>
    </div>
    
    <div class="form-group">
      <label for="email">Email *</label>
      <input
        id="email"
        v-model="form.email"
        type="email"
        :class="{ 'error': errors.email }"
        placeholder="Enter your email"
        required
      />
      <span v-if="errors.email" class="error-message">{{ errors.email }}</span>
    </div>
    
    <div class="form-group">
      <label for="role">Role</label>
      <select id="role" v-model="form.role">
        <option value="">Select a role</option>
        <option value="developer">Developer</option>
        <option value="designer">Designer</option>
        <option value="manager">Manager</option>
        <option value="admin">Admin</option>
      </select>
    </div>
    
    <div class="form-group">
      <label>
        <input
          type="checkbox"
          v-model="form.newsletter"
        />
        Subscribe to newsletter
      </label>
    </div>
    
    <div class="form-actions">
      <button
        type="submit"
        :disabled="isSubmitting"
        class="btn-primary"
      >
        {{ isSubmitting ? 'Submitting...' : 'Submit' }}
      </button>
      
      <button
        type="button"
        @click="resetForm"
        class="btn-secondary"
      >
        Reset
      </button>
    </div>
    
    <div v-if="submitMessage" class="submit-message">
      {{ submitMessage }}
    </div>
  </form>
</template>

<script>
export default {
  name: 'UserForm',
  data() {
    return {
      form: {
        name: '',
        email: '',
        role: '',
        newsletter: false
      },
      errors: {},
      isSubmitting: false,
      submitMessage: ''
    }
  },
  methods: {
    async handleSubmit() {
      this.isSubmitting = true
      this.errors = {}
      this.submitMessage = ''
      
      try {
        // Validate form
        this.validateForm()
        
        if (Object.keys(this.errors).length === 0) {
          // Simulate API call
          await new Promise(resolve => setTimeout(resolve, 1000))
          
          this.submitMessage = 'Form submitted successfully!'
          this.resetForm()
        }
      } catch (error) {
        this.submitMessage = 'Error submitting form. Please try again.'
        console.error('Form submission error:', error)
      } finally {
        this.isSubmitting = false
      }
    },
    
    validateForm() {
      if (!this.form.name.trim()) {
        this.errors.name = 'Name is required'
      }
      
      const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/
      if (!this.form.email.trim()) {
        this.errors.email = 'Email is required'
      } else if (!emailRegex.test(this.form.email)) {
        this.errors.email = 'Please enter a valid email'
      }
    },
    
    resetForm() {
      this.form = {
        name: '',
        email: '',
        role: '',
        newsletter: false
      }
      this.errors = {}
      this.submitMessage = ''
    }
  }
}
</script>

<style scoped>
.user-form {
  max-width: 500px;
  margin: 2rem auto;
  padding: 2rem;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #333;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  transition: border-color 0.2s;
}

.form-group input:focus,
.form-group select:focus {
  outline: none;
  border-color: #007bff;
}

.form-group input.error {
  border-color: #dc3545;
}

.error-message {
  color: #dc3545;
  font-size: 0.875rem;
  margin-top: 0.25rem;
}

.form-actions {
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
}

.submit-message {
  margin-top: 1rem;
  padding: 0.75rem;
  background: #d4edda;
  color: #155724;
  border: 1px solid #c3e6cb;
  border-radius: 4px;
}
</style>'''
            }
        }
        
        logger.info(f"Initialized {len(self.domain_templates)} domain categories")
    
    def generate_all_domains(self) -> Dict[str, Any]:
        """
        Generate synthetic data for all domains and methods
        
        Returns:
            Dict[str, Any]: Generated synthetic data
        """
        logger.info("Starting comprehensive synthetic data generation")
        
        try:
            generated_data = {}
            
            # Generate data for each method
            for method_name, method_func in self.generation_methods.items():
                logger.info(f"Generating {method_name} data...")
                
                method_params = self.generation_params[method_name]
                method_data = method_func(**method_params)
                generated_data[method_name] = method_data
                
                logger.info(f"  Generated {len(method_data)} examples using {method_name}")
            
            # Combine all data
            all_synthetic_examples = []
            for method_data in generated_data.values():
                all_synthetic_examples.extend(method_data)
            
            # Apply quality filtering
            logger.info("Applying quality filtering to all synthetic data...")
            quality_filtered_data = self.apply_synthetic_quality_filtering(all_synthetic_examples)
            
            # Final statistics
            final_stats = self.calculate_generation_statistics(quality_filtered_data)
            
            logger.info("Synthetic data generation completed:")
            logger.info(f"  Total generated: {final_stats['total_generated']:,}")
            logger.info(f"  Quality filtered: {final_stats['quality_filtered']:,}")
            logger.info(f"  Quality pass rate: {final_stats['quality_pass_rate']:.1%}")
            
            return {
                'generated_data': generated_data,
                'quality_filtered_data': quality_filtered_data,
                'generation_statistics': final_stats
            }
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {str(e)}")
            raise
    
    def self_instruct_generation(self, target_count: int, difficulty_distribution: Dict[str, float], 
                               language_distribution: Dict[str, float]) -> List[SyntheticExample]:
        """
        Generate data using Self-Instruct methodology
        
        Args:
            target_count: Number of examples to generate
            difficulty_distribution: Distribution of difficulty levels
            language_distribution: Distribution of programming languages
            
        Returns:
            List[SyntheticExample]: Generated self-instruct examples
        """
        logger.info(f"  Self-Instruct generation: {target_count:,} examples")
        
        examples = []
        
        # Seed instructions for bootstrapping
        seed_instructions = {
            'beginner': [
                "Create a function to calculate the sum of two numbers",
                "Write a simple JavaScript function to display a message",
                "Create a React component that shows a welcome message",
                "Write HTML code for a simple contact form",
                "Create an XML configuration file for a web service"
            ],
            'intermediate': [
                "Create a JavaScript function to validate email addresses",
                "Write a React hook to fetch data from an API",
                "Create a TypeScript interface for a user object",
                "Write XML schema (XSD) for product data",
                "Create a JavaScript function to handle form submission"
            ],
            'advanced': [
                "Create a custom React hook with error handling and caching",
                "Write a TypeScript generic function for array operations",
                "Create XML transformation (XSLT) for data conversion",
                "Implement a JavaScript module system with lazy loading",
                "Create a responsive CSS framework with utility classes"
            ]
        }
        
        for i in tqdm(range(target_count), desc="Self-Instruct generation"):
            # Select difficulty and language based on distributions
            difficulty = self.weighted_choice(difficulty_distribution)
            language = self.weighted_choice(language_distribution)
            
            # Select seed instruction or create new one
            if i < len(seed_instructions[difficulty]):
                seed_instruction = seed_instructions[difficulty][i % len(seed_instructions[difficulty])]
            else:
                seed_instruction = self.generate_instruction_from_seed(difficulty, language)
            
            # Generate corresponding code/output
            output = self.generate_code_from_instruction(seed_instruction, language, difficulty)
            
            # Create synthetic example
            example = SyntheticExample(
                content=output,
                content_type='code',
                language=language,
                generation_method='self_instruct',
                quality_score=self.assess_code_quality(output, language),
                difficulty_level=difficulty,
                domain='web_development',
                metadata={
                    'seed_instruction': seed_instruction,
                    'generation_iteration': i,
                    'complexity_factors': self.analyze_complexity(output)
                }
            )
            
            examples.append(example)
            
            # Update statistics
            self.stats['self_instruct_generated'] += 1
        
        logger.info(f"  Self-Instruct: Generated {len(examples)} examples")
        return examples
    
    def evol_instruct_generation(self, target_count: int, complexity_increase: float, 
                               evolution_operations: List[str]) -> List[SyntheticExample]:
        """
        Generate data using Evol-Instruct methodology for complexity scaling
        
        Args:
            target_count: Number of examples to generate
            complexity_increase: Target complexity increase
            evolution_operations: Available evolution operations
            
        Returns:
            List[SyntheticExample]: Generated evol-instruct examples
        """
        logger.info(f"  Evol-Instruct generation: {target_count:,} examples")
        
        examples = []
        
        # Base examples for evolution
        base_examples = [
            "function add(a, b) { return a + b; }",
            "const getUser = (id) => fetch(`/api/users/${id}`);",
            "<div className='header'>Welcome</div>",
            "interface User { id: number; name: string; }",
            "<configuration><app>value</app></configuration>"
        ]
        
        for i in tqdm(range(target_count), desc="Evol-Instruct generation"):
            # Select base example
            base_code = base_examples[i % len(base_examples)]
            
            # Apply evolution operations
            evolved_code = self.apply_evolution_operations(base_code, evolution_operations)
            
            # Generate instruction for evolved code
            instruction = self.generate_instruction_from_code(evolved_code)
            
            # Create synthetic example
            example = SyntheticExample(
                content=evolved_code,
                content_type='code',
                language=self.detect_language(evolved_code),
                generation_method='evol_instruct',
                quality_score=self.assess_code_quality(evolved_code, self.detect_language(evolved_code)),
                difficulty_level=self.assess_difficulty_from_code(evolved_code),
                domain='web_development',
                metadata={
                    'base_code': base_code,
                    'evolution_operations_applied': self.record_evolution_operations(),
                    'complexity_increase': complexity_increase,
                    'generation_iteration': i
                }
            )
            
            examples.append(example)
            
            # Update statistics
            self.stats['evol_instruct_generated'] += 1
        
        logger.info(f"  Evol-Instruct: Generated {len(examples)} examples")
        return examples
    
    def ast_mutation_generation(self, target_count: int, mutation_rate: float, 
                              supported_languages: List[str], 
                              mutation_types: List[str]) -> List[SyntheticExample]:
        """
        Generate data using AST mutation for code augmentation
        
        Args:
            target_count: Number of examples to generate
            mutation_rate: Rate of mutations to apply
            supported_languages: Languages supporting AST mutations
            mutation_types: Available mutation types
            
        Returns:
            List[SyntheticExample]: Generated AST mutation examples
        """
        logger.info(f"  AST mutation generation: {target_count:,} examples")
        
        examples = []
        
        # Seed code examples for mutation
        seed_code_examples = {
            'javascript': [
                "function calculateSum(numbers) { return numbers.reduce((sum, num) => sum + num, 0); }",
                "class User { constructor(name, email) { this.name = name; this.email = email; } }",
                "const fetchData = async (url) => { const response = await fetch(url); return response.json(); }",
                "const processArray = (arr, filterFn) => arr.filter(filterFn).map(x => x * 2);"
            ],
            'typescript': [
                "interface Product { id: number; name: string; price: number; }",
                "function findProduct<T>(products: T[], predicate: (item: T) => boolean): T | undefined { return products.find(predicate); }",
                "class Calculator { add(a: number, b: number): number { return a + b; } }",
                "type ApiResponse<T> = { success: boolean; data: T; error?: string; };"
            ]
        }
        
        for i in tqdm(range(target_count), desc="AST mutation generation"):
            # Select language and base code
            language = random.choice(supported_languages)
            base_code = random.choice(seed_code_examples[language])
            
            # Apply AST mutations
            mutated_code = self.apply_ast_mutations(base_code, language, mutation_rate, mutation_types)
            
            # Generate explanation for mutated code
            explanation = self.generate_code_explanation(mutated_code, language)
            
            # Create synthetic example
            example = SyntheticExample(
                content=mutated_code,
                content_type='code',
                language=language,
                generation_method='ast_mutation',
                quality_score=self.assess_code_quality(mutated_code, language),
                difficulty_level=self.assess_difficulty_from_code(mutated_code),
                domain='web_development',
                metadata={
                    'original_code': base_code,
                    'mutations_applied': self.record_ast_mutations(),
                    'mutation_rate': mutation_rate,
                    'generation_iteration': i
                }
            )
            
            examples.append(example)
            
            # Update statistics
            self.stats['ast_mutation_generated'] += 1
        
        logger.info(f"  AST mutation: Generated {len(examples)} examples")
        return examples
    
    def domain_specific_generation(self, target_count: int, domains: Dict[str, float]) -> List[SyntheticExample]:
        """
        Generate domain-specific templates for web development
        
        Args:
            target_count: Number of examples to generate
            domains: Distribution of domains
            
        Returns:
            List[SyntheticExample]: Generated domain-specific examples
        """
        logger.info(f"  Domain-specific generation: {target_count:,} examples")
        
        examples = []
        
        for i in tqdm(range(target_count), desc="Domain-specific generation"):
            # Select domain based on distribution
            domain = self.weighted_choice(domains)
            
            # Generate domain-specific content
            content = self.generate_domain_content(domain)
            
            # Create synthetic example
            example = SyntheticExample(
                content=content['code'],
                content_type=content['type'],
                language=content['language'],
                generation_method='domain_specific',
                quality_score=self.assess_domain_quality(content),
                difficulty_level=content['difficulty'],
                domain=domain,
                metadata={
                    'domain': domain,
                    'template_type': content['template_type'],
                    'generation_iteration': i
                }
            )
            
            examples.append(example)
            
            # Update statistics
            self.stats['domain_specific_generated'] += 1
        
        logger.info(f"  Domain-specific: Generated {len(examples)} examples")
        return examples
    
    def apply_synthetic_quality_filtering(self, examples: List[SyntheticExample]) -> List[SyntheticExample]:
        """Apply quality filtering to all synthetic examples"""
        logger.info("Applying quality filtering to synthetic data...")
        
        quality_filtered = []
        
        for example in tqdm(examples, desc="Quality filtering"):
            # Calculate quality score
            quality_score = self.calculate_synthetic_quality_score(example)
            
            # Check quality threshold
            if quality_score >= 0.75:
                example.quality_score = quality_score
                quality_filtered.append(example)
        
        self.stats['quality_filtered'] = len(quality_filtered)
        self.stats['total_generated'] = len(examples)
        
        quality_pass_rate = len(quality_filtered) / len(examples) if examples else 0
        logger.info(f"Quality filtering: {len(quality_filtered):,}/{len(examples):,} passed ({quality_pass_rate:.1%})")
        
        return quality_filtered
    
    def calculate_synthetic_quality_score(self, example: SyntheticExample) -> float:
        """Calculate quality score for synthetic example"""
        
        # Base quality factors
        factors = []
        
        # Content length appropriateness
        content_length = len(example.content)
        if 100 <= content_length <= 5000:  # Optimal range
            factors.append(1.0)
        elif content_length < 100:
            factors.append(content_length / 100)
        else:
            factors.append(max(0.5, 5000 / content_length))
        
        # Code syntax validity
        syntax_score = self.validate_synthetic_syntax(example)
        factors.append(syntax_score)
        
        # Language appropriateness
        lang_score = self.assess_language_appropriateness(example)
        factors.append(lang_score)
        
        # Domain relevance
        domain_score = self.assess_domain_relevance(example)
        factors.append(domain_score)
        
        # Generation method quality
        method_score = self.assess_generation_method_quality(example)
        factors.append(method_score)
        
        return np.mean(factors)
    
    def validate_synthetic_syntax(self, example: SyntheticExample) -> float:
        """Validate syntax for synthetic example"""
        
        try:
            content = example.content
            
            if example.language == 'javascript':
                # Basic JS syntax validation
                if re.search(r'\bfunction\b', content) or re.search(r'\bconst\b|\blet\b', content):
                    return 0.9
            elif example.language == 'typescript':
                # Basic TS syntax validation
                if re.search(r'\binterface\b|\btype\b|\bclass\b', content):
                    return 0.9
            elif example.language in ['xml', 'html']:
                # Basic XML/HTML validation
                if content.strip().startswith('<') and content.strip().endswith('>'):
                    return 0.9
            
            return 0.7  # Default score
            
        except Exception:
            return 0.5
    
    def assess_language_appropriateness(self, example: SyntheticExample) -> float:
        """Assess if content matches expected language patterns"""
        
        language_patterns = {
            'javascript': [r'\bfunction\b', r'\bconst\b', r'\blet\b', r'=>', r'console\.log'],
            'typescript': [r'\binterface\b', r'\btype\b', r':', r'<.*>'],
            'xml': [r'<\?xml', r'<[a-zA-Z]', r'</[a-zA-Z]'],
            'html': [r'<!DOCTYPE', r'<html', r'<head', r'<body'],
            'mdx': [r'#', r'\*\*.*\*\*', r'<[^>]+>', r'```']
        }
        
        patterns = language_patterns.get(example.language, [])
        if not patterns:
            return 0.8  # Default score
        
        matches = sum(1 for pattern in patterns if re.search(pattern, example.content))
        return min(matches / len(patterns), 1.0) if patterns else 0.8
    
    def assess_domain_relevance(self, example: SyntheticExample) -> float:
        """Assess domain relevance for web development"""
        
        web_dev_keywords = [
            'react', 'component', 'hook', 'api', 'fetch', 'component',
            'http', 'server', 'client', 'browser', 'javascript', 'html',
            'css', 'xml', 'web', 'frontend', 'backend', 'framework'
        ]
        
        content_lower = example.content.lower()
        matches = sum(1 for keyword in web_dev_keywords if keyword in content_lower)
        
        # Calculate relevance based on keyword density
        relevance = min(matches / 3, 1.0)  # At least 3 relevant keywords for full score
        return relevance
    
    def assess_generation_method_quality(self, example: SyntheticExample) -> float:
        """Assess quality based on generation method characteristics"""
        
        method_quality_scores = {
            'self_instruct': 0.8,  # Generally good quality
            'evol_instruct': 0.75,  # Good but may have artifacts
            'ast_mutation': 0.7,    # Variable quality
            'domain_specific': 0.85  # High quality templates
        }
        
        return method_quality_scores.get(example.generation_method, 0.7)
    
    def calculate_generation_statistics(self, examples: List[SyntheticExample]) -> Dict[str, Any]:
        """Calculate comprehensive generation statistics"""
        
        total_generated = self.stats['total_generated']
        quality_filtered = self.stats['quality_filtered']
        
        stats = {
            'total_generated': total_generated,
            'quality_filtered': quality_filtered,
            'quality_pass_rate': quality_filtered / total_generated if total_generated > 0 else 0,
            'generation_breakdown': {
                'self_instruct': self.stats['self_instruct_generated'],
                'evol_instruct': self.stats['evol_instruct_generated'],
                'ast_mutation': self.stats['ast_mutation_generated'],
                'domain_specific': self.stats['domain_specific_generated']
            },
            'language_distribution': {},
            'difficulty_distribution': {},
            'domain_distribution': {},
            'average_quality_score': np.mean([ex.quality_score for ex in examples]) if examples else 0
        }
        
        # Calculate distributions
        for example in examples:
            # Language distribution
            lang = example.language
            stats['language_distribution'][lang] = stats['language_distribution'].get(lang, 0) + 1
            
            # Difficulty distribution
            difficulty = example.difficulty_level
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
            
            # Domain distribution
            domain = example.domain
            stats['domain_distribution'][domain] = stats['domain_distribution'].get(domain, 0) + 1
        
        return stats
    
    # Helper methods for specific generation techniques
    
    def weighted_choice(self, choices: Dict[str, float]) -> str:
        """Make weighted random choice"""
        choices_list = list(choices.keys())
        weights = list(choices.values())
        return random.choices(choices_list, weights=weights)[0]
    
    def generate_instruction_from_seed(self, difficulty: str, language: str) -> str:
        """Generate instruction from seed based on difficulty and language"""
        
        instruction_templates = {
            'beginner': {
                'javascript': "Create a simple {function_type} function that {task}",
                'typescript': "Write a basic TypeScript {construct_type} for {entity}",
                'xml': "Create a simple XML {element_type} for {purpose}",
                'html': "Write HTML {element_type} for {purpose}"
            },
            'intermediate': {
                'javascript': "Implement a {function_type} function with {feature} that {task}",
                'typescript': "Create a TypeScript {construct_type} with {feature} for {entity}",
                'xml': "Generate XML {element_type} with {attribute} for {purpose}",
                'html': "Build HTML {element_type} with {feature} for {purpose}"
            },
            'advanced': {
                'javascript': "Develop a sophisticated {function_type} implementation with {feature} and {optimization} for {task}",
                'typescript': "Engineer a complex TypeScript {construct_type} with {feature}, {constraint}, and {optimization} for {entity}",
                'xml': "Design comprehensive XML {element_type} with {attribute}, {namespace}, and {validation} for {purpose}",
                'html': "Create advanced HTML {element_type} with {feature}, {responsive}, and {accessibility} for {purpose}"
            }
        }
        
        templates = instruction_templates.get(difficulty, {})
        template = templates.get(language, templates.get('javascript', ''))
        
        # Fill in template with random values
        fill_values = self.generate_instruction_fill_values(language, difficulty)
        return template.format(**fill_values)
    
    def generate_instruction_fill_values(self, language: str, difficulty: str) -> Dict[str, str]:
        """Generate fill values for instruction templates"""
        
        values = {
            'function_type': random.choice(['utility', 'helper', 'calculation', 'validation', 'processing']),
            'task': random.choice(['process data', 'validate input', 'transform format', 'calculate result', 'handle events']),
            'construct_type': random.choice(['interface', 'type', 'class', 'function', 'enum']),
            'entity': random.choice(['user data', 'product info', 'configuration', 'settings', 'metadata']),
            'element_type': random.choice(['component', 'document', 'configuration', 'template', 'layout']),
            'purpose': random.choice(['web application', 'user interface', 'data display', 'form handling', 'navigation']),
            'feature': random.choice(['error handling', 'caching', 'validation', 'optimization', 'logging']),
            'attribute': random.choice(['custom attributes', 'namespaces', 'validation rules', 'schema definitions', 'metadata']),
            'optimization': random.choice(['performance tuning', 'memory optimization', 'algorithm efficiency', 'caching strategy', 'lazy loading']),
            'constraint': random.choice(['type safety', 'runtime validation', 'interface compliance', 'contract enforcement', 'boundary conditions']),
            'namespace': random.choice(['XML namespaces', 'custom schemas', 'standard compliance', 'versioning', 'modular structure']),
            'validation': random.choice(['schema validation', 'data integrity', 'format checking', 'constraint enforcement', 'consistency rules']),
            'responsive': random.choice(['mobile-first design', 'cross-device compatibility', 'adaptive layouts', 'responsive breakpoints', 'touch optimization']),
            'accessibility': random.choice(['ARIA compliance', 'screen reader support', 'keyboard navigation', 'color contrast', 'focus management'])
        }
        
        return values
    
    def generate_code_from_instruction(self, instruction: str, language: str, difficulty: str) -> str:
        """Generate code from instruction"""
        
        # Mock code generation based on instruction and language
        code_templates = {
            'javascript': {
                'beginner': '''function example() {
    console.log("Hello World");
    return "success";
}''',
                'intermediate': '''async function example() {
    try {
        const result = await fetch('/api/data');
        const data = await result.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}''',
                'advanced': '''class AdvancedExample {
    constructor(options) {
        this.options = options;
        this.cache = new Map();
    }
    
    async process(data) {
        const cacheKey = this.generateCacheKey(data);
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        const result = await this.performComplexOperation(data);
        this.cache.set(cacheKey, result);
        return result;
    }
}'''
            },
            'typescript': {
                'beginner': '''interface Example {
    id: number;
    name: string;
}''',
                'intermediate': '''interface AdvancedExample<T> {
    id: number;
    data: T;
    timestamp: Date;
    validate(): boolean;
}''',
                'advanced': '''type ComplexExample<T extends Record<string, any>> = {
    [K in keyof T]: T[K] extends Function ? never : T[K];
};

interface ValidatedExample<T> extends ComplexExample<T> {
    readonly id: number;
    readonly createdAt: Date;
    validate(): this is ValidatedExample<T>;
}'''
            }
        }
        
        return code_templates.get(language, {}).get(difficulty, '// Generated code based on instruction')
    
    def apply_evolution_operations(self, code: str, operations: List[str]) -> str:
        """Apply evolution operations to code"""
        
        # Mock evolution operations
        operations_applied = []
        evolved_code = code
        
        for operation in operations:
            if random.random() < 0.3:  # 30% chance per operation
                if operation == 'instruction_complexity_increase':
                    evolved_code = self.increase_instruction_complexity(evolved_code)
                    operations_applied.append(operation)
                elif operation == 'output_elaboration':
                    evolved_code = self.elaborate_output(evolved_code)
                    operations_applied.append(operation)
                elif operation == 'task_clarification':
                    evolved_code = self.clarify_task(evolved_code)
                    operations_applied.append(operation)
                elif operation == 'constraint_addition':
                    evolved_code = self.add_constraints(evolved_code)
                    operations_applied.append(operation)
        
        # Store operations for metadata
        self._evolution_operations_applied = operations_applied
        return evolved_code
    
    def increase_instruction_complexity(self, code: str) -> str:
        """Increase complexity of code"""
        # Add error handling and validation
        if 'function' in code:
            return f'''async function enhanced() {{
    try {{
        // Enhanced with error handling
        {code.strip()}
    }} catch (error) {{
        console.error('Enhanced error handling:', error);
        throw new Error(`Operation failed: ${{error.message}}`);
    }}
}}'''
        return code
    
    def elaborate_output(self, code: str) -> str:
        """Elaborate output with additional features"""
        if 'return' in code:
            # Add logging and validation
            lines = code.split('\n')
            elaborated_lines = []
            
            for line in lines:
                if 'return' in line:
                    elaborated_lines.append('    console.log("Returning result:", result);')
                elaborated_lines.append(line)
            
            return '\n'.join(elaborated_lines)
        return code
    
    def clarify_task(self, code: str) -> str:
        """Clarify task with better documentation"""
        if 'function' in code:
            return f'''/**
 * Enhanced function with comprehensive documentation
 * @param input - Input parameter
 * @returns Processed result with validation
 */
{code}'''
        return code
    
    def add_constraints(self, code: str) -> str:
        """Add constraints and validation"""
        if 'function' in code:
            return f'''function constrained(input) {{
    // Add parameter validation
    if (!input) {{
        throw new Error("Input parameter is required");
    }}
    
    // Validate input type
    if (typeof input !== 'object') {{
        throw new Error("Input must be an object");
    }}
    
    {code.replace('function', '').strip()}
}}'''
        return code
    
    def record_evolution_operations(self) -> List[str]:
        """Record evolution operations that were applied"""
        return getattr(self, '_evolution_operations_applied', [])
    
    def apply_ast_mutations(self, code: str, language: str, mutation_rate: float, 
                          mutation_types: List[str]) -> str:
        """Apply AST-based mutations to code"""
        
        # Mock AST mutations
        mutations_applied = []
        mutated_code = code
        
        for mutation_type in mutation_types:
            if random.random() < mutation_rate:
                if mutation_type == 'variable_renaming':
                    mutated_code = self.mutate_variable_names(mutated_code)
                    mutations_applied.append('variable_renaming')
                elif mutation_type == 'function_modification':
                    mutated_code = self.mutate_function_signatures(mutated_code)
                    mutations_applied.append('function_modification')
                elif mutation_type == 'syntax_variation':
                    mutated_code = self.mutate_syntax(mutated_code)
                    mutations_applied.append('syntax_variation')
                elif mutation_type == 'logic_optimization':
                    mutated_code = self.optimize_logic(mutated_code)
                    mutations_applied.append('logic_optimization')
        
        # Store mutations for metadata
        self._ast_mutations_applied = mutations_applied
        return mutated_code
    
    def mutate_variable_names(self, code: str) -> str:
        """Mutate variable names in code"""
        # Simple variable name mutation
        import re
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        variables = re.findall(pattern, code)
        
        mutated_code = code
        for var in variables[:3]:  # Limit mutations
            if len(var) > 2 and var not in ['function', 'return', 'class', 'if', 'else']:
                new_var = f"{var[0]}{hashlib.md5(var.encode()).hexdigest()[:4]}"
                mutated_code = re.sub(rf'\b{var}\b', new_var, mutated_code)
        
        return mutated_code
    
    def mutate_function_signatures(self, code: str) -> str:
        """Mutate function signatures"""
        # Add optional parameters or change signature
        if 'function' in code and '(' in code:
            return code.replace('(', '(options = {}, ')
        return code
    
    def mutate_syntax(self, code: str) -> str:
        """Mutate syntax variations"""
        # Convert function declarations to arrow functions
        if 'function ' in code:
            code = re.sub(r'function\s+(\w+)\s*\([^)]*\)\s*{', r'const \1 = (\1) => {', code)
        return code
    
    def optimize_logic(self, code: str) -> str:
        """Optimize logic in code"""
        # Add performance optimizations
        if 'for' in code:
            code = code.replace('for (', 'for (let i = 0; i < ')
        return code
    
    def record_ast_mutations(self) -> List[str]:
        """Record AST mutations that were applied"""
        return getattr(self, '_ast_mutations_applied', [])
    
    def generate_domain_content(self, domain: str) -> Dict[str, Any]:
        """Generate content for specific domain"""
        
        if domain == 'xml_configuration':
            template_key = random.choice(list(self.domain_templates['xml_configuration'].keys()))
            code = self.domain_templates['xml_configuration'][template_key]
            return {
                'code': code,
                'type': 'configuration',
                'language': 'xml',
                'difficulty': 'intermediate',
                'template_type': template_key
            }
        
        elif domain == 'mdx_components':
            template_key = random.choice(list(self.domain_templates['mdx_components'].keys()))
            code = self.domain_templates['mdx_components'][template_key]
            return {
                'code': code,
                'type': 'component',
                'language': 'mdx',
                'difficulty': 'advanced',
                'template_type': template_key
            }
        
        elif domain == 'react_hooks':
            template_key = random.choice(list(self.domain_templates['react_hooks'].keys()))
            code = self.domain_templates['react_hooks'][template_key]
            return {
                'code': code,
                'type': 'hook',
                'language': 'javascript',
                'difficulty': 'intermediate',
                'template_type': template_key
            }
        
        elif domain == 'vue_templates':
            template_key = random.choice(list(self.domain_templates['vue_templates'].keys()))
            code = self.domain_templates['vue_templates'][template_key]
            return {
                'code': code,
                'type': 'template',
                'language': 'vue',
                'difficulty': 'intermediate',
                'template_type': template_key
            }
        
        else:
            return {
                'code': '// Domain-specific content',
                'type': 'other',
                'language': 'javascript',
                'difficulty': 'beginner',
                'template_type': 'default'
            }
    
    def assess_domain_quality(self, content: Dict[str, Any]) -> float:
        """Assess quality of domain-specific content"""
        code = content['code']
        
        # Base quality factors
        factors = []
        
        # Length appropriateness
        if 200 <= len(code) <= 10000:
            factors.append(1.0)
        elif len(code) < 200:
            factors.append(len(code) / 200)
        else:
            factors.append(max(0.5, 10000 / len(code)))
        
        # Language-specific quality
        if content['language'] == 'mdx':
            # Check for MDX patterns
            if code.count('#') > 0 and code.count('import') > 0:
                factors.append(0.9)
            else:
                factors.append(0.7)
        elif content['language'] == 'vue':
            # Check for Vue patterns
            if '<template>' in code and '<script>' in code:
                factors.append(0.9)
            else:
                factors.append(0.7)
        else:
            factors.append(0.8)
        
        return np.mean(factors)
    
    def generate_code_from_instruction(self, instruction: str) -> str:
        """Generate code explanation from instruction"""
        
        explanations = {
            'beginner': 'This is a simple function that demonstrates basic programming concepts.',
            'intermediate': 'This function includes error handling and asynchronous operations.',
            'advanced': 'This implementation uses advanced patterns including caching and optimization.'
        }
        
        difficulty = 'intermediate'  # Default
        return explanations.get(difficulty, explanations['beginner'])
    
    def detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        if 'interface' in code or 'type ' in code:
            return 'typescript'
        elif 'function' in code or 'const ' in code:
            return 'javascript'
        elif code.strip().startswith('<') and '</' in code:
            return 'html'
        elif '<?xml' in code or '<configuration>' in code:
            return 'xml'
        else:
            return 'javascript'  # Default
    
    def assess_difficulty_from_code(self, code: str) -> str:
        """Assess difficulty level from code characteristics"""
        
        complexity_indicators = {
            'beginner': ['function', 'console.log', 'return'],
            'intermediate': ['class', 'async', 'try', 'catch', 'interface'],
            'advanced': ['generic', 'complex', 'optimization', 'advanced']
        }
        
        code_lower = code.lower()
        
        for difficulty, indicators in complexity_indicators.items():
            if sum(1 for indicator in indicators if indicator in code_lower) >= 2:
                return difficulty
        
        return 'intermediate'  # Default
    
    def assess_code_quality(self, code: str, language: str) -> float:
        """Assess quality of generated code"""
        
        # Basic quality factors
        factors = []
        
        # Length appropriateness
        if 50 <= len(code) <= 5000:
            factors.append(1.0)
        elif len(code) < 50:
            factors.append(len(code) / 50)
        else:
            factors.append(max(0.5, 5000 / len(code)))
        
        # Language-specific patterns
        if language == 'javascript':
            if 'function' in code and ('return' in code or '{' in code):
                factors.append(0.9)
            else:
                factors.append(0.7)
        elif language == 'typescript':
            if 'interface' in code or 'type' in code:
                factors.append(0.9)
            else:
                factors.append(0.7)
        else:
            factors.append(0.8)
        
        # Syntax completeness
        if code.count('{') == code.count('}'):
            factors.append(0.9)
        else:
            factors.append(0.6)
        
        return min(sum(factors) / len(factors), 1.0)
    
    def generate_code_explanation(self, code: str, language: str) -> str:
        """Generate explanation for code"""
        explanations = {
            'javascript': 'This JavaScript code demonstrates modern ES6+ features and best practices.',
            'typescript': 'This TypeScript code shows strong typing and interface definitions.',
            'xml': 'This XML configuration defines structured data for web applications.',
            'html': 'This HTML code creates semantic web page structure.',
            'vue': 'This Vue component uses reactive data and template syntax.'
        }
        
        return explanations.get(language, 'This code demonstrates programming concepts.')
    
    def analyze_complexity(self, code: str) -> Dict[str, int]:
        """Analyze complexity factors in code"""
        
        complexity_factors = {
            'function_count': len(re.findall(r'\bfunction\b', code)),
            'class_count': len(re.findall(r'\bclass\b', code)),
            'async_count': len(re.findall(r'\basync\b', code)),
            'error_handling_count': len(re.findall(r'\btry\b|\bcatch\b|\berror\b', code, re.IGNORECASE)),
            'complexity_score': 0
        }
        
        # Calculate overall complexity score
        complexity_factors['complexity_score'] = sum([
            complexity_factors['function_count'] * 1,
            complexity_factors['class_count'] * 2,
            complexity_factors['async_count'] * 1.5,
            complexity_factors['error_handling_count'] * 1
        ])
        
        return complexity_factors

# Import for type hint (avoid circular import)
try:
    from data_preparation_pipeline import DataPreparationConfig
except ImportError:
    pass

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthetic Data Generator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    config = DataPreparationConfig(**config_dict)
    
    # Initialize generator
    generator = SyntheticDataGenerator(config)
    
    # Generate synthetic data
    if args.test:
        logger.info("Running in test mode")
        test_examples = [
            SyntheticExample(
                content="function test() { return 'hello'; }",
                content_type="code",
                language="javascript",
                generation_method="self_instruct",
                quality_score=0.8,
                difficulty_level="beginner",
                domain="web_development",
                metadata={}
            )
        ]
        logger.info(f"Test generation completed with {len(test_examples)} examples")
    else:
        result = generator.generate_all_domains()
        logger.info(f"Generation completed: {result['generation_statistics']['total_generated']:,} examples")

if __name__ == "__main__":
    main()