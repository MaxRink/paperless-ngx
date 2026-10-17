from contextvars import Context
from unittest import mock

from auditlog.models import LogEntry
from django.contrib.auth.models import User
from django.db.models import Model
from django.db.models import QuerySet
from django.test import TestCase
from rest_framework.test import APIClient
from treenode.models import TreeNodeModel

from documents.models import Tag
from documents.models import _tag_tree_updates


class TestTagMetadataBatching(TestCase):
    def setUp(self):
        self.parent = Tag.objects.create(name="Parent")
        self.child = Tag.objects.create(name="Child", tn_parent=self.parent)

    def test_metadata_save_keeps_structure_and_refreshes_cached_names(self):
        self.assertEqual(self.child.get_parent().name, "Parent")
        with mock.patch.object(TreeNodeModel, "update_tree") as rebuild:
            self.parent.name = "Renamed parent"
            self.parent.matching_algorithm = Tag.MATCH_NONE
            self.parent.save()
        rebuild.assert_not_called()
        self.child.refresh_from_db()
        self.assertEqual(self.child.get_parent().name, "Renamed parent")
        self.assertEqual(self.child.get_ancestors_pks(), [str(self.parent.pk)])
        self.parent.refresh_from_db()
        self.assertEqual(self.parent.matching_algorithm, Tag.MATCH_NONE)

    def test_reparent_and_create_still_rebuild(self):
        other = Tag.objects.create(name="Other")
        self.child.tn_parent = other
        self.child.save()
        self.child.refresh_from_db()
        self.assertEqual(self.child.get_ancestors_pks(), [str(other.pk)])
        created = Tag.objects.create(name="Grandchild", tn_parent=self.child)
        self.assertEqual(
            created.get_ancestors_pks(),
            [str(other.pk), str(self.child.pk)],
        )

    def test_bulk_delete_rebuilds_once_and_keeps_survivor(self):
        sibling = Tag.objects.create(name="Sibling", tn_parent=self.parent)
        calls = []
        original = TreeNodeModel.update_tree.__func__

        def rebuild(cls):
            calls.append(cls)
            return original(cls)

        with mock.patch.object(TreeNodeModel, "update_tree", classmethod(rebuild)):
            Tag.objects.filter(pk__in=[self.child.pk, sibling.pk]).delete()
        self.assertEqual(calls, [Tag])
        self.parent.refresh_from_db()
        self.assertEqual(self.parent.get_children_pks(), [])
        self.assertTrue(Tag.objects.filter(pk=self.parent.pk).exists())

    def test_failed_save_and_delete_restore_context(self):
        with mock.patch.object(Model, "save", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                self.parent.save()
        self.assertTrue(_tag_tree_updates.get())
        with mock.patch.object(QuerySet, "delete", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                Tag.objects.filter(pk=self.child.pk).delete()
        self.assertTrue(_tag_tree_updates.get())

    def test_suppression_is_context_local(self):
        token = _tag_tree_updates.set(False)
        try:
            self.assertTrue(Context().run(_tag_tree_updates.get))
        finally:
            _tag_tree_updates.reset(token)

    def test_native_api_metadata_patch_keeps_normal_audit_path(self):
        user = User.objects.create_user(username="synthetic-admin", is_superuser=True)
        client = APIClient()
        client.force_authenticate(user)
        with mock.patch.object(TreeNodeModel, "update_tree") as rebuild:
            response = client.patch(
                f"/api/tags/{self.parent.pk}/",
                {"matching_algorithm": Tag.MATCH_NONE},
                format="json",
            )
        self.assertEqual(response.status_code, 200)
        rebuild.assert_not_called()
        self.parent.refresh_from_db()
        self.assertEqual(self.parent.matching_algorithm, Tag.MATCH_NONE)
        event = LogEntry.objects.filter(
            object_pk=str(self.parent.pk),
            content_type__model="tag",
            action=LogEntry.Action.UPDATE,
        ).latest("timestamp")
        self.assertEqual(int(event.changes_dict["matching_algorithm"][1]), 0)

    def test_stale_metadata_instance_does_not_erase_a_new_child(self):
        stale = Tag.objects.get(pk=self.parent.pk)
        added = Tag.objects.create(name="Added", tn_parent=self.parent)
        stale.matching_algorithm = Tag.MATCH_NONE
        stale.save()
        stale.refresh_from_db()
        self.assertIn(str(added.pk), stale.get_children_pks())

    def test_empty_bulk_delete_does_not_rebuild(self):
        with mock.patch.object(TreeNodeModel, "update_tree") as rebuild:
            Tag.objects.filter(pk=-1).delete()
        rebuild.assert_not_called()
