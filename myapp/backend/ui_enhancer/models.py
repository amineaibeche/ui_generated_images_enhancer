from django.db import models

class ui_enhancer(models.Model):
    title=models.CharField(max_length=150)
    description=models.CharField(max_length=500)
    completed=models.BooleanField(default=False)

    # string representation of the class
    def __str__(self):

        #it will return the title
        return self.title 
    

from django.db import models

class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    prompt = models.TextField(blank=True, null=True)
    processed_image = models.ImageField(upload_to='processed/', blank=True, null=True)
    quality_score = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id}"
