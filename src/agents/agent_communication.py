"""
AgentVault™ Agent Communication Hub
Inter-agent communication and message routing system
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict

import aioredis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import websockets

from ..database.models import Agent
from ..cache.distributed_cache import DistributedCache
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class DeliveryMode(str, Enum):
    """Message delivery modes"""
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    ORDERED = "ordered"


@dataclass
class Message:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    
    # Routing information
    source_agent_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    target_group: Optional[str] = None  # For group messaging
    routing_key: Optional[str] = None   # For topic-based routing
    
    # Content
    subject: str = ""
    body: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Tracking
    retry_count: int = 0
    max_retries: int = 3
    acknowledged: bool = False
    delivered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'type': self.type,
            'priority': self.priority,
            'delivery_mode': self.delivery_mode,
            'source_agent_id': self.source_agent_id,
            'target_agent_id': self.target_agent_id,
            'target_group': self.target_group,
            'routing_key': self.routing_key,
            'subject': self.subject,
            'body': self.body,
            'headers': self.headers,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        # Parse timestamps
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'expires_at' in data and data['expires_at'] and isinstance(data['expires_at'], str):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        
        return cls(**data)


@dataclass
class MessageHandler:
    """Handler for processing messages"""
    pattern: str  # Routing pattern (e.g., "agent.*.command", "system.#")
    handler: Callable[[Message], asyncio.coroutine]
    filter: Optional[Callable[[Message], bool]] = None
    priority: int = 0  # Higher priority handlers are called first


class MessageBroker:
    """
    Central message broker for agent communication
    Supports multiple transport mechanisms
    """
    
    def __init__(
        self,
        redis_url: str,
        kafka_config: Optional[Dict[str, Any]] = None,
        monitoring: Optional[AdvancedMonitoringSystem] = None
    ):
        self.redis_url = redis_url
        self.kafka_config = kafka_config or {}
        self.monitoring = monitoring
        
        # Connections
        self.redis: Optional[aioredis.Redis] = None
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None
        
        # Message routing
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._agent_channels: Dict[str, str] = {}  # agent_id -> channel
        self._group_members: Dict[str, Set[str]] = defaultdict(set)  # group -> agent_ids
        
        # Message tracking
        self._pending_messages: Dict[str, Message] = {}
        self._message_callbacks: Dict[str, Callable] = {}
        
        # Background tasks
        self._consumer_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize message broker connections"""
        logger.info("Initializing Message Broker")
        
        # Connect to Redis
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
        # Initialize Kafka if configured
        if self.kafka_config.get('bootstrap_servers'):
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await self.kafka_producer.start()
            
            self.kafka_consumer = AIOKafkaConsumer(
                'agentvault.messages',
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                group_id='agentvault-broker',
                value_deserializer=lambda v: json.loads(v.decode())
            )
            await self.kafka_consumer.start()
            
            # Start Kafka consumer
            self._consumer_tasks.append(
                asyncio.create_task(self._kafka_consumer_loop())
            )
        
        # Start Redis subscriber
        self._consumer_tasks.append(
            asyncio.create_task(self._redis_subscriber_loop())
        )
        
        # Start message processor
        self._consumer_tasks.append(
            asyncio.create_task(self._message_processor_loop())
        )
        
        logger.info("Message Broker initialized")
    
    async def shutdown(self):
        """Shutdown message broker"""
        logger.info("Shutting down Message Broker")
        
        self._shutdown_event.set()
        
        # Stop consumers
        await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        
        # Close connections
        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        
        logger.info("Message Broker shutdown complete")
    
    async def send_message(
        self,
        message: Message,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Send a message
        Returns response for request/response pattern
        """
        # Validate message
        if message.expires_at and message.expires_at < datetime.utcnow():
            raise ValueError("Message has already expired")
        
        # Set default expiry
        if not message.expires_at:
            message.expires_at = datetime.utcnow() + timedelta(minutes=5)
        
        # Route message
        if message.target_agent_id:
            await self._route_to_agent(message)
        elif message.target_group:
            await self._route_to_group(message)
        elif message.routing_key:
            await self._route_by_key(message)
        else:
            raise ValueError("Message must have target_agent_id, target_group, or routing_key")
        
        # Record metrics
        if self.monitoring:
            self.monitoring.record_custom_metric(
                "agent_message_sent",
                1,
                labels={
                    'type': message.type,
                    'priority': message.priority,
                    'source_agent': message.source_agent_id or 'system'
                }
            )
        
        # Wait for response if request/response pattern
        if message.type == MessageType.REQUEST and timeout:
            return await self._wait_for_response(message.id, timeout)
        
        return None
    
    async def broadcast_message(
        self,
        message: Message,
        agent_filter: Optional[Callable[[str], bool]] = None
    ):
        """Broadcast message to all agents (with optional filter)"""
        message.type = MessageType.BROADCAST
        
        # Get all agent channels
        agent_ids = list(self._agent_channels.keys())
        
        # Apply filter if provided
        if agent_filter:
            agent_ids = [aid for aid in agent_ids if agent_filter(aid)]
        
        # Send to each agent
        tasks = []
        for agent_id in agent_ids:
            msg_copy = Message.from_dict(message.to_dict())
            msg_copy.id = str(uuid.uuid4())  # New ID for each copy
            msg_copy.target_agent_id = agent_id
            tasks.append(self._route_to_agent(msg_copy))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def register_handler(
        self,
        pattern: str,
        handler: Callable[[Message], asyncio.coroutine],
        filter: Optional[Callable[[Message], bool]] = None,
        priority: int = 0
    ):
        """Register a message handler"""
        handler_obj = MessageHandler(
            pattern=pattern,
            handler=handler,
            filter=filter,
            priority=priority
        )
        
        # Extract base pattern for indexing
        base_pattern = pattern.split('.')[0]
        self._handlers[base_pattern].append(handler_obj)
        
        # Sort by priority
        self._handlers[base_pattern].sort(key=lambda h: h.priority, reverse=True)
    
    def unregister_handler(self, pattern: str, handler: Callable):
        """Unregister a message handler"""
        base_pattern = pattern.split('.')[0]
        self._handlers[base_pattern] = [
            h for h in self._handlers[base_pattern]
            if h.pattern != pattern or h.handler != handler
        ]
    
    async def register_agent(self, agent_id: str, channel: Optional[str] = None):
        """Register an agent for communication"""
        if not channel:
            channel = f"agent.{agent_id}"
        
        self._agent_channels[agent_id] = channel
        
        # Subscribe to agent channel in Redis
        if self.redis:
            # Implementation depends on Redis pub/sub setup
            pass
        
        logger.info(f"Registered agent {agent_id} on channel {channel}")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self._agent_channels:
            channel = self._agent_channels[agent_id]
            del self._agent_channels[agent_id]
            
            # Remove from all groups
            for group_members in self._group_members.values():
                group_members.discard(agent_id)
            
            logger.info(f"Unregistered agent {agent_id}")
    
    async def join_group(self, agent_id: str, group: str):
        """Add agent to a group"""
        self._group_members[group].add(agent_id)
        logger.info(f"Agent {agent_id} joined group {group}")
    
    async def leave_group(self, agent_id: str, group: str):
        """Remove agent from a group"""
        self._group_members[group].discard(agent_id)
        if not self._group_members[group]:
            del self._group_members[group]
        logger.info(f"Agent {agent_id} left group {group}")
    
    async def get_agent_groups(self, agent_id: str) -> List[str]:
        """Get groups an agent belongs to"""
        return [
            group for group, members in self._group_members.items()
            if agent_id in members
        ]
    
    # Private routing methods
    
    async def _route_to_agent(self, message: Message):
        """Route message to specific agent"""
        if message.target_agent_id not in self._agent_channels:
            raise ValueError(f"Unknown agent: {message.target_agent_id}")
        
        channel = self._agent_channels[message.target_agent_id]
        
        # Use appropriate transport
        if message.priority == MessagePriority.CRITICAL:
            # Use Redis for critical messages (lower latency)
            await self._send_via_redis(channel, message)
        elif message.delivery_mode == DeliveryMode.EXACTLY_ONCE:
            # Use Kafka for exactly-once delivery
            await self._send_via_kafka(channel, message)
        else:
            # Default to Redis
            await self._send_via_redis(channel, message)
    
    async def _route_to_group(self, message: Message):
        """Route message to group members"""
        if message.target_group not in self._group_members:
            logger.warning(f"Empty or unknown group: {message.target_group}")
            return
        
        # Send to each group member
        tasks = []
        for agent_id in self._group_members[message.target_group]:
            msg_copy = Message.from_dict(message.to_dict())
            msg_copy.id = str(uuid.uuid4())
            msg_copy.target_agent_id = agent_id
            msg_copy.target_group = None  # Clear group to avoid loops
            tasks.append(self._route_to_agent(msg_copy))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _route_by_key(self, message: Message):
        """Route message by routing key pattern"""
        # Find matching handlers
        matching_handlers = []
        
        for base_pattern, handlers in self._handlers.items():
            if message.routing_key.startswith(base_pattern):
                for handler in handlers:
                    if self._matches_pattern(message.routing_key, handler.pattern):
                        if not handler.filter or handler.filter(message):
                            matching_handlers.append(handler)
        
        # Sort by priority
        matching_handlers.sort(key=lambda h: h.priority, reverse=True)
        
        # Execute handlers
        for handler in matching_handlers:
            try:
                await handler.handler(message)
            except Exception as e:
                logger.error(f"Handler error for {handler.pattern}: {str(e)}")
    
    def _matches_pattern(self, routing_key: str, pattern: str) -> bool:
        """Check if routing key matches pattern"""
        # Simple pattern matching (* = single segment, # = multiple segments)
        pattern_parts = pattern.split('.')
        key_parts = routing_key.split('.')
        
        i = j = 0
        while i < len(pattern_parts) and j < len(key_parts):
            if pattern_parts[i] == '#':
                # Match rest of the key
                return True
            elif pattern_parts[i] == '*' or pattern_parts[i] == key_parts[j]:
                i += 1
                j += 1
            else:
                return False
        
        return i == len(pattern_parts) and j == len(key_parts)
    
    # Transport methods
    
    async def _send_via_redis(self, channel: str, message: Message):
        """Send message via Redis pub/sub"""
        if not self.redis:
            raise RuntimeError("Redis not initialized")
        
        # Store message for reliability
        if message.delivery_mode != DeliveryMode.FIRE_AND_FORGET:
            await self.redis.setex(
                f"msg:{message.id}",
                int((message.expires_at - datetime.utcnow()).total_seconds()),
                json.dumps(message.to_dict())
            )
        
        # Publish message
        await self.redis.publish(channel, json.dumps(message.to_dict()))
    
    async def _send_via_kafka(self, topic: str, message: Message):
        """Send message via Kafka"""
        if not self.kafka_producer:
            # Fallback to Redis
            await self._send_via_redis(topic, message)
            return
        
        # Send to Kafka
        await self.kafka_producer.send(
            topic='agentvault.messages',
            key=topic.encode(),
            value=message.to_dict(),
            headers=[
                ('message_id', message.id.encode()),
                ('priority', message.priority.encode())
            ]
        )
    
    async def _wait_for_response(
        self,
        request_id: str,
        timeout: float
    ) -> Optional[Message]:
        """Wait for response to a request"""
        response_event = asyncio.Event()
        response_message = None
        
        def response_handler(message: Message):
            nonlocal response_message
            if message.correlation_id == request_id:
                response_message = message
                response_event.set()
        
        # Register callback
        self._message_callbacks[request_id] = response_handler
        
        try:
            # Wait for response
            await asyncio.wait_for(response_event.wait(), timeout)
            return response_message
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out after {timeout}s")
            return None
        finally:
            # Clean up callback
            self._message_callbacks.pop(request_id, None)
    
    # Consumer loops
    
    async def _redis_subscriber_loop(self):
        """Subscribe to Redis channels"""
        if not self.redis:
            return
        
        try:
            # Create subscription
            channel_patterns = ['agent.*', 'group.*', 'broadcast.*']
            
            while not self._shutdown_event.is_set():
                # This is a simplified implementation
                # In production, use proper Redis pub/sub
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Redis subscriber error: {str(e)}")
    
    async def _kafka_consumer_loop(self):
        """Consume messages from Kafka"""
        if not self.kafka_consumer:
            return
        
        try:
            async for msg in self.kafka_consumer:
                if self._shutdown_event.is_set():
                    break
                
                try:
                    # Parse message
                    message = Message.from_dict(msg.value)
                    
                    # Route message
                    await self._process_message(message)
                    
                except Exception as e:
                    logger.error(f"Kafka message processing error: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Kafka consumer error: {str(e)}")
    
    async def _message_processor_loop(self):
        """Process pending messages"""
        while not self._shutdown_event.is_set():
            try:
                # Process retries
                for msg_id, message in list(self._pending_messages.items()):
                    if message.expires_at < datetime.utcnow():
                        # Message expired
                        del self._pending_messages[msg_id]
                        continue
                    
                    if not message.acknowledged and message.retry_count < message.max_retries:
                        # Retry message
                        message.retry_count += 1
                        await self._route_message(message)
                
            except Exception as e:
                logger.error(f"Message processor error: {str(e)}")
            
            await asyncio.sleep(5)  # Process every 5 seconds
    
    async def _process_message(self, message: Message):
        """Process incoming message"""
        # Check for response callback
        if message.type == MessageType.RESPONSE and message.correlation_id:
            callback = self._message_callbacks.get(message.correlation_id)
            if callback:
                callback(message)
                return
        
        # Route by pattern
        await self._route_by_key(message)
    
    async def _route_message(self, message: Message):
        """Route a message (used for retries)"""
        if message.target_agent_id:
            await self._route_to_agent(message)
        elif message.target_group:
            await self._route_to_group(message)
        elif message.routing_key:
            await self._route_by_key(message)


class AgentCommunicationHub:
    """
    High-level communication hub for agents
    Provides simplified API over MessageBroker
    """
    
    def __init__(
        self,
        broker: MessageBroker,
        cache: DistributedCache
    ):
        self.broker = broker
        self.cache = cache
        
        # Agent registry
        self._registered_agents: Dict[str, Dict[str, Any]] = {}
        
        # Request tracking
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # WebSocket connections for real-time communication
        self._websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
    
    async def register_agent(
        self,
        agent: Agent,
        capabilities: List[str],
        groups: Optional[List[str]] = None
    ):
        """Register an agent with the communication hub"""
        agent_id = str(agent.id)
        
        # Register with broker
        await self.broker.register_agent(agent_id)
        
        # Join groups
        if groups:
            for group in groups:
                await self.broker.join_group(agent_id, group)
        
        # Store agent info
        self._registered_agents[agent_id] = {
            'agent': agent,
            'capabilities': capabilities,
            'groups': groups or [],
            'registered_at': datetime.utcnow(),
            'status': 'online'
        }
        
        # Broadcast agent online event
        await self.broker.send_message(Message(
            type=MessageType.EVENT,
            routing_key='agent.online',
            source_agent_id=agent_id,
            subject='Agent Online',
            body={
                'agent_id': agent_id,
                'capabilities': capabilities
            }
        ))
        
        logger.info(f"Agent {agent_id} registered with capabilities: {capabilities}")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id not in self._registered_agents:
            return
        
        # Broadcast offline event
        await self.broker.send_message(Message(
            type=MessageType.EVENT,
            routing_key='agent.offline',
            source_agent_id=agent_id,
            subject='Agent Offline',
            body={'agent_id': agent_id}
        ))
        
        # Unregister from broker
        await self.broker.unregister_agent(agent_id)
        
        # Clean up
        del self._registered_agents[agent_id]
        
        # Close WebSocket if exists
        if agent_id in self._websocket_connections:
            await self._websocket_connections[agent_id].close()
            del self._websocket_connections[agent_id]
        
        logger.info(f"Agent {agent_id} unregistered")
    
    async def send_request(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        body: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Send request from one agent to another"""
        message = Message(
            type=MessageType.REQUEST,
            source_agent_id=from_agent,
            target_agent_id=to_agent,
            subject=subject,
            body=body
        )
        
        response = await self.broker.send_message(message, timeout=timeout)
        
        if response:
            return response.body
        return None
    
    async def send_command(
        self,
        to_agent: str,
        command: str,
        parameters: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Send command to an agent"""
        message = Message(
            type=MessageType.COMMAND,
            target_agent_id=to_agent,
            subject=f"command.{command}",
            body=parameters,
            priority=priority
        )
        
        try:
            await self.broker.send_message(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send command {command} to {to_agent}: {str(e)}")
            return False
    
    async def broadcast_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        source_agent: Optional[str] = None
    ):
        """Broadcast an event to all interested agents"""
        message = Message(
            type=MessageType.EVENT,
            routing_key=f"event.{event_type}",
            source_agent_id=source_agent,
            subject=event_type,
            body=data
        )
        
        await self.broker.send_message(message)
    
    async def subscribe_to_events(
        self,
        agent_id: str,
        event_patterns: List[str],
        handler: Callable[[Dict[str, Any]], asyncio.coroutine]
    ):
        """Subscribe agent to event patterns"""
        for pattern in event_patterns:
            self.broker.register_handler(
                pattern=f"event.{pattern}",
                handler=lambda msg: handler(msg.body),
                filter=lambda msg: msg.source_agent_id != agent_id  # Don't receive own events
            )
    
    async def create_channel(
        self,
        channel_name: str,
        members: List[str]
    ) -> str:
        """Create a communication channel for multiple agents"""
        channel_id = f"channel.{channel_name}"
        
        # Add members to group
        for member in members:
            await self.broker.join_group(member, channel_id)
        
        # Store channel info
        await self.cache.set(
            f"channel:{channel_id}",
            json.dumps({
                'name': channel_name,
                'members': members,
                'created_at': datetime.utcnow().isoformat()
            }),
            ttl=86400  # 24 hours
        )
        
        return channel_id
    
    async def send_to_channel(
        self,
        channel_id: str,
        sender: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Send message to a channel"""
        msg = Message(
            type=MessageType.NOTIFICATION,
            source_agent_id=sender,
            target_group=channel_id,
            subject='channel_message',
            body={
                'message': message,
                'data': data or {},
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        await self.broker.send_message(msg)
    
    async def establish_websocket(
        self,
        agent_id: str,
        websocket: websockets.WebSocketServerProtocol
    ):
        """Establish WebSocket connection for real-time communication"""
        self._websocket_connections[agent_id] = websocket
        
        # Send connection established message
        await websocket.send(json.dumps({
            'type': 'connected',
            'agent_id': agent_id,
            'timestamp': datetime.utcnow().isoformat()
        }))
        
        # Handle incoming messages
        try:
            async for message in websocket:
                data = json.loads(message)
                
                # Convert to Message and route
                msg = Message(
                    type=MessageType(data.get('type', 'request')),
                    source_agent_id=agent_id,
                    target_agent_id=data.get('target'),
                    subject=data.get('subject', ''),
                    body=data.get('body', {})
                )
                
                await self.broker.send_message(msg)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket closed for agent {agent_id}")
        finally:
            if agent_id in self._websocket_connections:
                del self._websocket_connections[agent_id]
    
    async def send_realtime_update(
        self,
        agent_id: str,
        update_type: str,
        data: Dict[str, Any]
    ):
        """Send real-time update to agent via WebSocket"""
        if agent_id in self._websocket_connections:
            websocket = self._websocket_connections[agent_id]
            
            await websocket.send(json.dumps({
                'type': 'update',
                'update_type': update_type,
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            }))
    
    def get_online_agents(self) -> List[str]:
        """Get list of online agents"""
        return [
            agent_id for agent_id, info in self._registered_agents.items()
            if info['status'] == 'online'
        ]
    
    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get capabilities of an agent"""
        if agent_id in self._registered_agents:
            return self._registered_agents[agent_id]['capabilities']
        return []
    
    async def find_agents_with_capability(self, capability: str) -> List[str]:
        """Find agents with specific capability"""
        return [
            agent_id for agent_id, info in self._registered_agents.items()
            if capability in info['capabilities'] and info['status'] == 'online'
        ]
    
    async def heartbeat_check(self):
        """Send heartbeat to all agents and check responses"""
        online_agents = self.get_online_agents()
        
        # Send heartbeat to each agent
        heartbeat_futures = {}
        
        for agent_id in online_agents:
            message = Message(
                type=MessageType.HEARTBEAT,
                target_agent_id=agent_id,
                subject='heartbeat',
                body={'timestamp': datetime.utcnow().isoformat()}
            )
            
            future = asyncio.create_task(
                self.broker.send_message(message, timeout=5.0)
            )
            heartbeat_futures[agent_id] = future
        
        # Wait for responses
        results = await asyncio.gather(*heartbeat_futures.values(), return_exceptions=True)
        
        # Update agent status
        for agent_id, result in zip(heartbeat_futures.keys(), results):
            if isinstance(result, Exception) or result is None:
                # Agent didn't respond
                if agent_id in self._registered_agents:
                    self._registered_agents[agent_id]['status'] = 'offline'
                    logger.warning(f"Agent {agent_id} failed heartbeat check")
            else:
                # Agent responded
                if agent_id in self._registered_agents:
                    self._registered_agents[agent_id]['status'] = 'online'