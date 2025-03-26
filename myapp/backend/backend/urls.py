from django.contrib import admin

# add include to the path
from django.urls import path, include

# import views from todo
from ui_enhancer import views

# import routers from the REST framework
# it is necessary for routing
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static

# create a router object
router = routers.DefaultRouter()

# register the router
router.register(r'tasks',views.ui_enhancerView, 'task')

urlpatterns = [
    path('admin/', admin.site.urls),

    # add another path to the url patterns
    # when you visit the localhost:8000/api
    # you should be routed to the django Rest framework
    path('api/', include(router.urls)),
    path('api/enhance/', views.enhance_view, name='enhance'),
    path('api/evaluate/', views.evaluate_view, name='evaluate'),


]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
