
import uuid

# Django Imports
from django.db import models
from django.utils.translation import gettext_lazy as _


def upload_path(instance, filename):
    return '{0}/{1}'.format(instance.uuid_identifier, filename)


class Workflow(models.Model):
    uuid_identifier = models.UUIDField(editable=False, default=uuid.uuid4, unique=True)
    index = models.FileField(upload_to=upload_path, blank=True, null=True, help_text=_("Insert the calculated index."))
    new_index = models.FileField(upload_to=upload_path, blank=True, null=True, help_text=_("Insert the calculated new index.") )
    training_vectors = models.CharField(max_length=255, blank=True, help_text=_("Insert the local path of the calculated training_vectors."))
    vocab_list = models.FileField(upload_to=upload_path, blank=True, null=True, help_text=_("Insert the calculated vocabulary list."))
    topic_predictions = models.FileField(upload_to=upload_path, blank=True, null=True, help_text=_("Insert topic predictions csv"))
    training_book_ids = models.TextField(blank=True, help_text=_("Insert the book ids for this flow separated by a comma."))
    ntm_predictor_endpoint = models.TextField(blank=True, help_text=_("Insert the NTM predictor endpoint."))
    knn_predictor_endpoint = models.TextField(blank=True, help_text=_("Insert the KNN predictor endpoint."))
    number_books = models.IntegerField(default=0, blank=True, help_text=_("Amount of books for this workflow."))
    s3_paths = models.JSONField(blank=True, null=True, help_text=_("Field saved for a small dictionary of S3 paths."))
    processing_times = models.JSONField(blank=True, null=True, default=dict, help_text=_("Field saved for keeping track of time it take to process data."))
    booklist = models.JSONField(blank=True, null=True, default=dict, help_text=_("List of books in the workflow."))

    def __str__(self):
        return f"Workflow: {self.uuid_identifier} - {self.number_books} books."

    class Meta:
        verbose_name = "Workflow"
        verbose_name_plural = "Workflows"


class Title(models.Model):
    identifier = models.CharField(max_length=20, help_text=_("Input the book identifier."))
    publisher = models.CharField(_("Publisher"),blank=True, max_length=120, help_text=_("Input the publisher associated to the book."))
    theme = models.CharField(_("Theme"), max_length=100, blank=True, help_text=_("Input the theme associated to the book."))
    name = models.CharField(_("Title name"),blank=True, max_length=250, help_text=_("Input the name given to the resource."))
    complete_text = models.TextField(_("Raw Text"), blank=True, help_text=_("Complete text of the book."))
    parsed_tokens = models.TextField(_("Tokens"), blank=True, help_text=_("Input tokens separated by a comma"))
    vector_file = models.FileField(upload_to="vectors_file", blank=True, null=True, help_text=_("Field to save a vector"))
    number_pages = models.IntegerField(_("Number of pages"), default=0, help_text=_("Input the amount of pages of the book."))
    training_book = models.BooleanField(_("Is for training?"), default=False, help_text=_("Is the book used for training?"))
    cover = models.URLField(_("Cover"), blank=True, null=True, help_text=_("Set the cover url"))

    class Meta:
        verbose_name = "Ranked Title"
        verbose_name_plural = "Ranked Titles"

    def __str__(self):
        return f'Title: {self.identifier} - name: > {self.name}'


class Recommendation(models.Model):
    title = models.ForeignKey(Title, verbose_name=_("Current Title"), related_name="base_title", on_delete=models.CASCADE)
    recommendation = models.ForeignKey(Title, verbose_name=_("Recommendation"), related_name="recommendation", on_delete=models.CASCADE)
    order = models.IntegerField(default=0)
    average_rating = models.DecimalField(_("Rating"), max_digits=6, decimal_places=2, help_text="This is a calculated field, please don't modify")
    workflow = models.ForeignKey(Workflow, blank=True, null=True, verbose_name=_("Workflow"), on_delete=models.CASCADE, help_text=_("Select the workflow associated to this recommendation."))
    distance = models.CharField(_("Distance"), blank=True, max_length=20, help_text=_("This represents the distance between vectors."))

    class Meta:
        verbose_name = "Recommendation"
        verbose_name_plural = "Recommendations"

    def __str__(self):
        return f'Base: {self.title.name} - Recommendation: {self.recommendation.name}'


class RecommendationRating(models.Model):
    recommendation = models.ForeignKey(Recommendation, on_delete=models.CASCADE)
    rating = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Title Rating"
        verbose_name_plural = "Title Ratings"

    def __str__(self):
        return f'Title: {self.recommendation.id} - Rating: {self.rating}'
