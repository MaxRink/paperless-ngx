from unittest.mock import patch

from django.contrib.auth.models import Permission
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from documents.models import Document
from documents.models import Tag
from documents.permissions import annotate_document_count_by_ids


class TestTagDetailCountQueryset(TestCase):
    def setUp(self) -> None:
        self.owner = User.objects.create_user(username="tag-owner")
        self.other_user = User.objects.create_user(username="other-owner")
        view_tag = Permission.objects.get(
            codename="view_tag",
            content_type__app_label="documents",
        )
        self.owner.user_permissions.add(view_tag)
        self.client = APIClient()
        self.client.force_authenticate(user=self.owner)

        self.first_tag = Tag.objects.create(name="First", owner=self.owner)
        self.second_tag = Tag.objects.create(name="Second", owner=self.owner)
        self.hidden_tag = Tag.objects.create(name="Hidden", owner=self.other_user)

        visible_document = Document.objects.create(
            title="Visible document",
            owner=self.owner,
        )
        hidden_document = Document.objects.create(
            title="Hidden document",
            owner=self.other_user,
        )
        visible_document.tags.add(self.first_tag)
        hidden_document.tags.add(self.first_tag)

    def _capture_count_querysets(self, observed: list[list[int]]):
        def capture(queryset, *args, **kwargs):
            observed.append(list(queryset.values_list("pk", flat=True)))
            return annotate_document_count_by_ids(queryset, *args, **kwargs)

        return capture

    def test_detail_scopes_count_annotation_and_keeps_user_document_counts(
        self,
    ) -> None:
        observed: list[list[int]] = []
        with patch(
            "documents.permissions.annotate_document_count_by_ids",
            side_effect=self._capture_count_querysets(observed),
        ):
            response = self.client.get(f"/api/tags/{self.first_tag.pk}/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["document_count"], 1)
        self.assertTrue(observed)
        self.assertTrue(all(ids == [self.first_tag.pk] for ids in observed))

    def test_list_keeps_full_queryset_and_document_counts(self) -> None:
        observed: list[list[int]] = []
        with patch(
            "documents.permissions.annotate_document_count_by_ids",
            side_effect=self._capture_count_querysets(observed),
        ):
            response = self.client.get("/api/tags/?full_perms=true")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = {row["id"]: row for row in response.json()["results"]}
        self.assertEqual(
            set(results),
            {self.first_tag.pk, self.second_tag.pk},
        )
        self.assertEqual(results[self.first_tag.pk]["document_count"], 1)
        self.assertEqual(results[self.second_tag.pk]["document_count"], 0)
        self.assertTrue(observed)
        self.assertTrue(
            all(
                set(ids) == {self.first_tag.pk, self.second_tag.pk, self.hidden_tag.pk}
                for ids in observed
            ),
        )

    def test_invalid_and_invisible_detail_ids_remain_not_found(self) -> None:
        invalid = self.client.get("/api/tags/not-an-integer/")
        invisible = self.client.get(f"/api/tags/{self.hidden_tag.pk}/")

        self.assertEqual(invalid.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(invisible.status_code, status.HTTP_404_NOT_FOUND)
