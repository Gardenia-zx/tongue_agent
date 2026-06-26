CREATE TABLE IF NOT EXISTS agent_turn_record (
  tenant_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  canonical_request_hash TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  thread_epoch INTEGER NOT NULL,
  status TEXT NOT NULL,
  owner_id TEXT NULL,
  started_at TIMESTAMPTZ NULL,
  lease_until TIMESTAMPTZ NULL,
  completed_at TIMESTAMPTZ NULL,
  response_snapshot_encrypted TEXT NULL,
  response_ref JSONB NULL,
  response_hash TEXT NULL,
  java_ack_status TEXT NOT NULL DEFAULT 'PENDING',
  failure_reason TEXT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (tenant_id, turn_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_turn_thread_status
ON agent_turn_record (tenant_id, thread_id, thread_epoch, status);

CREATE TABLE IF NOT EXISTS agent_thread_epoch (
  tenant_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  current_epoch INTEGER NOT NULL,
  reset_reason TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (tenant_id, thread_id)
);

CREATE TABLE IF NOT EXISTS memory_inbox (
  event_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_ref JSONB NULL,
  failure_reason TEXT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
