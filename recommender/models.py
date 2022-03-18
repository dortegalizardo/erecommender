
# Django Imports
from django.db import models
from django.utils.translation import gettext_lazy as _


class Title(models.Model):
    identifier = models.CharField(max_length=20, help_text=_("Input the book identifier."))
    publisher = models.CharField(_("Publisher"),blank=True, max_length=120, help_text=_("Input the publisher associated to the book."))
    theme = models.CharField(_("Theme"), max_length=100, blank=True, help_text=_("Input the theme associated to the book."))
    name = models.CharField(_("Title name"),blank=True, max_length=250, help_text=_("Input the name given to the resource."))
    complete_text = models.TextField(_("Raw Text"), blank=True, help_text=_("Complete text of the book."))
    parsed_tokens = models.TextField(_("Tokens"), blank=True, help_text=_("Input tokens separated by a comma"))
    vector_file = models.FileField(upload_to="vectors_file", blank=True, null=True, help_text=_("Field to save a vector"))

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
