from django.shortcuts import render

# import view sets from the REST framework
from rest_framework import viewsets

# import the TodoSerializer from the serializer file
from .serializers import ui_enhancerSerializer

# import the Todo model from the models file
from .models import ui_enhancer

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import base64
import traceback
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json
from .models_inf import mos_align_models, mos_models , enhancemnt_models , error_map_align
import traceback

# create a class for the Todo model viewsets
class ui_enhancerView(viewsets.ModelViewSet):

    # create a serializer class and 
    # assign it to the TodoSerializer class
    serializer_class = ui_enhancerSerializer

    # define a variable and populate it 
    # with the Todo list objects
    queryset = ui_enhancer.objects.all()





from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import base64
import traceback
import os
from PIL import Image  # Import PIL for handling images
from django.conf import settings

from django.conf import settings

@csrf_exempt
def enhance_view(request):
    if request.method == 'POST':
        try:
            # Retrieve all parameters from the request
            image_file = request.FILES.get('image')
            prompt = request.POST.get('prompt', '')
            steps = int(request.POST.get('steps', 25))  # Default: 25
            quality = request.POST.get('quality', 'high')  # Default: high
            detail_level = int(request.POST.get('detail_level', 50))  # Default: 50
            guidance_scale = float(request.POST.get('guidance_scale', 7.5))  # Default: 7.5
            strength = float(request.POST.get('strength', 0.5))  # Default: 0.5
            perceptual_model = request.POST.get('perceptual_model', 'ResNet-18')  # Default: ResNet-18
            alignment_model = request.POST.get('alignment_model', 'AlignNet-1_t5small_without_cross_attention')  # Default
            enhancement_model = request.POST.get('enhancement_model', 'Prompt_enhancement')  # Default

            # Validate the image file
            if not image_file:
                return JsonResponse({'error': 'No image file provided'}, status=400)

            # Save the uploaded image temporarily
            path = default_storage.save('tmp/image.jpg', ContentFile(image_file.read()))
            temp_file = default_storage.path(path)
            print(f"Temporary file saved at: {temp_file}")
            print(prompt)

            # Call your enhancement function
            try:
                if perceptual_model == "ResNet-18":
                    evaluation_mos = mos_models.resnet_18_inference(temp_file)
                else:
                    evaluation_mos = mos_models.resnet_50_inference(temp_file)

                if alignment_model == "AlignNet-1_t5small_without_cross_attention":
                    alignment_mos = mos_align_models.inference_without_cross_attention(temp_file, prompt)
                elif alignment_model == "AlignNet-2_t5small_and_cross_attention":
                    alignment_mos = mos_align_models.inference_cross_attention_t5(temp_file, prompt)
                else:
                    alignment_mos = mos_align_models.inference_cross_attention_bert(temp_file, prompt)

                # Get the relative path of the enhanced image
                if(enhancement_model == "Prompt_enhancement"):    
                    enhanced_image_relative_path = enhancemnt_models.infer_prompt_ehancement(
                        temp_file,
                        prompt,
                        evaluation_mos,
                        alignment_mos,
                        50,
                        0.85,
                        0.99
                    )
                else : 
                    enhanced_image_relative_path = error_map_align.error_map_align_infer(temp_file, prompt ,evaluation_mos , alignment_mos,steps , strength , 0.85)

                # Construct the full URL for the enhanced image
                enhanced_image_url = request.build_absolute_uri(f"{settings.MEDIA_URL}{enhanced_image_relative_path}")
                print(f"Enhanced image URL: {enhanced_image_url}")
            except Exception as e:
                print(f"Error during enhancement: {str(e)}")
                traceback.print_exc()
                return JsonResponse({'error': f'Enhancement failed: {str(e)}'}, status=500)

            # Clean up the temporary files
            default_storage.delete(path)

            # Return the result with the enhanced image URL
            return JsonResponse({
                'enhanced_image_url': enhanced_image_url
            })

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def evaluate_view(request):
    print("try")
    if request.method == 'POST':
        print("in the if")
        try:
            # Retrieve the uploaded image and prompt

            image_file = request.FILES.get('image')
            prompt = request.POST.get('prompt', '')
            steps = request.POST.get('steps', 25)  # Default value: 25
            quality = request.POST.get('quality', 'high')  # Default value: high
            detail_level = request.POST.get('detail_level', 50)  # Default value: 50
            guidance_scale = request.POST.get('guidance_scale', 7.5)  # Default value: 7.5
            strength = request.POST.get('strength', 0.5)  # Default value: 0.5
            perceptual_model = request.POST.get('perceptual_model', 'ResNet-18')  # Default model
            alignment_model = request.POST.get('alignment_model', 'AlignNet-1_t5small_without_cross_attention')
            #get all the  sed parameters
            print(image_file)
            # Validate the image file
            if not image_file:
                return JsonResponse({'error': 'No image file provided'}, status=400)

            # Save the uploaded image temporarily
            path = default_storage.save('tmp/image.jpg', ContentFile(image_file.read()))
            temp_file = default_storage.path(path)
            print(f"Temporary file saved at: {temp_file}")
            print(alignment_model)

            # Call your evaluation functions
            try:
                #print(temp_file)
                if(perceptual_model=="ResNet-18"):
                    evaluation_mos = mos_models.resnet_18_inference(temp_file)
                else:
                    evaluation_mos = mos_models.resnet_50_inference(temp_file)

                if(alignment_model == "AlignNet-1_t5small_without_cross_attention"):
                    alignement_mos = mos_align_models.inference_without_cross_attention(temp_file, prompt)
                else:
                    if(alignment_model == "AlignNet-2_t5small_and_cross_attention"):
                        alignement_mos = mos_align_models.inference_cross_attention_t5(temp_file, prompt)
                    else:
                        alignement_mos = mos_align_models.inference_cross_attention_bert(temp_file, prompt)
    
                print(f"Evaluation MOS: {evaluation_mos}, Alignment MOS: {alignement_mos}")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                traceback.print_exc()
                return JsonResponse({'error': f'Evaluation failed: {str(e)}'}, status=500)

            # Return the result
            return JsonResponse({
                'perceptual_quality': evaluation_mos,
                'alignment_quality': alignement_mos
                # Add other metrics here
            })

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
