from django.contrib import admin

from .models import UploadedDataset


@admin.register(UploadedDataset)
class UploadedDatasetAdmin(admin.ModelAdmin):
    list_display  = ('original_name', 'problem_type', 'n_samples', 'n_features', 'uploaded_at')
    list_filter   = ('problem_type',)
    search_fields = ('original_name', 'filename')
    ordering      = ('-uploaded_at',)
