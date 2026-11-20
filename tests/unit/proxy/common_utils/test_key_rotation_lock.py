"""
Test distributed lock behavior for key rotation manager.

Verifies that PodLockManager is correctly used to prevent concurrent
key rotation across multiple pods in a distributed deployment.
"""

from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.common_utils.key_rotation_manager import KeyRotationManager


class TestKeyRotationLock:
    """Test distributed lock behavior in KeyRotationManager."""

    @pytest.mark.asyncio
    async def test_process_rotations_acquires_lock(self):
        """
        When PodLockManager is provided and lock is acquired,
        rotation logic should run normally.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()  # Redis is available
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        # Mock _find_keys_needing_rotation to return empty list (no keys to rotate)
        manager._find_keys_needing_rotation = AsyncMock(return_value=[])
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify lock was acquired with custom TTL
        mock_pod_lock_manager.acquire_lock.assert_called_once()
        call_kwargs = mock_pod_lock_manager.acquire_lock.call_args
        assert call_kwargs.kwargs["cronjob_id"] == "litellm_key_rotation_job"
        assert call_kwargs.kwargs["ttl"] >= 300  # At least 5 minutes

        # Verify rotation logic ran (cleanup + find keys called)
        manager._cleanup_expired_deprecated_keys.assert_called_once()
        manager._find_keys_needing_rotation.assert_called_once()

        # Verify lock was released
        mock_pod_lock_manager.release_lock.assert_called_once_with(
            cronjob_id="litellm_key_rotation_job",
        )

    @pytest.mark.asyncio
    async def test_process_rotations_skips_when_lock_held(self):
        """
        When lock is held by another pod, process_rotations() should
        return early without performing any rotation.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=False)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        manager._find_keys_needing_rotation = AsyncMock()
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify lock was attempted
        mock_pod_lock_manager.acquire_lock.assert_called_once()

        # Verify rotation logic was NOT executed
        manager._cleanup_expired_deprecated_keys.assert_not_called()
        manager._find_keys_needing_rotation.assert_not_called()

        # Verify lock was NOT released (since it was never acquired)
        mock_pod_lock_manager.release_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_rotations_releases_lock_on_success(self):
        """
        Lock should be released in the finally block after successful rotation.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        # Simulate finding and rotating a key successfully
        mock_key = LiteLLM_VerificationToken(
            token="test-token",
            auto_rotate=True,
            rotation_interval="30s",
            key_rotation_at=None,
            rotation_count=0,
            key_name="test-key",
        )
        manager._find_keys_needing_rotation = AsyncMock(return_value=[mock_key])
        manager._cleanup_expired_deprecated_keys = AsyncMock()
        manager._rotate_key = AsyncMock()

        await manager.process_rotations()

        # Verify rotation was performed
        manager._rotate_key.assert_called_once_with(mock_key)

        # Verify lock was released after success
        mock_pod_lock_manager.release_lock.assert_called_once_with(
            cronjob_id="litellm_key_rotation_job",
        )

    @pytest.mark.asyncio
    async def test_process_rotations_releases_lock_on_error(self):
        """
        Lock should be released in the finally block even if rotation
        throws an exception.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        # Simulate an error during cleanup
        manager._cleanup_expired_deprecated_keys = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        await manager.process_rotations()

        # Verify lock was still released despite the error
        mock_pod_lock_manager.release_lock.assert_called_once_with(
            cronjob_id="litellm_key_rotation_job",
        )

    @pytest.mark.asyncio
    async def test_process_rotations_works_without_lock_manager(self):
        """
        When pod_lock_manager=None, rotation should run normally
        without any lock logic (backward compat / single-pod mode).
        """
        mock_prisma_client = AsyncMock()

        # No pod_lock_manager provided (default None)
        manager = KeyRotationManager(mock_prisma_client)

        manager._find_keys_needing_rotation = AsyncMock(return_value=[])
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify rotation logic ran normally
        manager._cleanup_expired_deprecated_keys.assert_called_once()
        manager._find_keys_needing_rotation.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_rotations_works_without_redis_cache(self):
        """
        When pod_lock_manager exists but redis_cache is None (no Redis configured),
        rotation should run normally without locking.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = None  # No Redis available

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        manager._find_keys_needing_rotation = AsyncMock(return_value=[])
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify lock was NOT attempted (no Redis)
        mock_pod_lock_manager.acquire_lock.assert_not_called()

        # Verify rotation logic still ran
        manager._cleanup_expired_deprecated_keys.assert_called_once()
        manager._find_keys_needing_rotation.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_rotations_handles_none_lock_result(self):
        """
        When acquire_lock returns None (edge case), it should be treated
        as lock NOT acquired, and rotation should be skipped.
        """
        mock_prisma_client = AsyncMock()
        mock_pod_lock_manager = MagicMock()
        mock_pod_lock_manager.redis_cache = MagicMock()
        mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=None)
        mock_pod_lock_manager.release_lock = AsyncMock()

        manager = KeyRotationManager(
            mock_prisma_client, pod_lock_manager=mock_pod_lock_manager
        )

        manager._find_keys_needing_rotation = AsyncMock()
        manager._cleanup_expired_deprecated_keys = AsyncMock()

        await manager.process_rotations()

        # Verify rotation logic was NOT executed (None treated as False via `or False`)
        manager._cleanup_expired_deprecated_keys.assert_not_called()
        manager._find_keys_needing_rotation.assert_not_called()

        # Verify lock was NOT released (lock_acquired is False)
        mock_pod_lock_manager.release_lock.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(("serve_last_good_entries", "reads_from_writer"), [(True, True), (False, False)])
async def test_a_rotation_run_picks_its_due_keys_from_the_writer_only_while_last_good_entries_are_served(
    serve_last_good_entries: bool, reads_from_writer: bool
):
    """Each rotation revokes its old token strictly and the run is outside any request, so while last good entries are
    served it pins its own reads to the writer: a key due on the writer is rotated though the lagging read replica
    does not list it yet. With the setting off the run reads from the replica as before."""
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    due_key: Final = LiteLLM_VerificationToken(
        token="due-token", auto_rotate=True, rotation_interval="30d", key_rotation_at=None, key_name="due-key"
    )
    writer: Final = PrismaWrapper(
        original_prisma=SimpleNamespace(
            litellm_verificationtoken=SimpleNamespace(find_many=AsyncMock(return_value=[due_key]), create=AsyncMock()),
            litellm_deprecatedverificationtoken=SimpleNamespace(delete_many=AsyncMock(return_value=0)),
        )
    )
    # The replica lags: it does not list the key the writer holds as due
    reader: Final = PrismaWrapper(
        original_prisma=SimpleNamespace(
            litellm_verificationtoken=SimpleNamespace(find_many=AsyncMock(return_value=[]), create=AsyncMock())
        )
    )
    prisma: Final = SimpleNamespace(db=RoutingPrismaWrapper(writer=writer, reader=reader), writer_db=writer)
    manager: Final = KeyRotationManager(prisma)  # pyright: ignore[reportArgumentType]  # a fake for PrismaClient
    rotate_key: Final = AsyncMock()
    manager._rotate_key = rotate_key
    cache: Final = UserApiKeyCache(serve_last_good_entries=serve_last_good_entries)

    with patch(  # test-quality-ok: the run reads the setting off this proxy_server global; no injection seam
        "litellm.proxy.proxy_server.user_api_key_cache", cache
    ):
        await manager.process_rotations()

    rotated: Final = [awaited.args[0] for awaited in rotate_key.await_args_list]
    assert rotated == ([due_key] if reads_from_writer else [])


@pytest.mark.asyncio
async def test_a_rotation_run_with_a_plain_dual_cache_as_the_auth_cache_still_rotates_its_due_keys():
    """The auth cache can be a plain DualCache, which keeps no last good entries: the run pins nothing, reads its due
    keys where it always did, and rotates them, rather than stopping on the setting only the cache's subclass
    carries."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    due_key: Final = LiteLLM_VerificationToken(
        token="due-token", auto_rotate=True, rotation_interval="30d", key_rotation_at=None, key_name="due-key"
    )
    writer: Final = PrismaWrapper(
        original_prisma=SimpleNamespace(
            litellm_verificationtoken=SimpleNamespace(find_many=AsyncMock(return_value=[]), create=AsyncMock()),
            litellm_deprecatedverificationtoken=SimpleNamespace(delete_many=AsyncMock(return_value=0)),
        )
    )
    reader: Final = PrismaWrapper(
        original_prisma=SimpleNamespace(
            litellm_verificationtoken=SimpleNamespace(find_many=AsyncMock(return_value=[due_key]), create=AsyncMock())
        )
    )
    prisma: Final = SimpleNamespace(db=RoutingPrismaWrapper(writer=writer, reader=reader), writer_db=writer)
    manager: Final = KeyRotationManager(prisma)  # pyright: ignore[reportArgumentType]  # a fake for PrismaClient
    rotated: Final[list[LiteLLM_VerificationToken]] = []  # mutable-ok: records each key the run rotates

    async def rotate_key(key: LiteLLM_VerificationToken) -> None:
        rotated.append(key)

    with (
        patch.object(manager, "_rotate_key", rotate_key),  # test-quality-ok: records the keys the run rotates
        patch(  # test-quality-ok: the run reads the setting off this proxy_server global; no injection seam
            "litellm.proxy.proxy_server.user_api_key_cache", DualCache()
        ),
    ):
        await manager.process_rotations()

    assert rotated == [due_key]
