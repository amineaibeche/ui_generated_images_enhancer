# import serializers from the REST framework
from rest_framework import serializers

# import the todo data model
from .models import ui_enhancer

# create a serializer class
class ui_enhancerSerializer(serializers.ModelSerializer):

    # create a meta class
    class Meta:
        model = ui_enhancer
        fields = ('id', 'title','description','completed')
