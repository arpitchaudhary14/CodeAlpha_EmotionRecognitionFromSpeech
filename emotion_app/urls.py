from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('search/', views.search_view, name='search'),
    path('model/', views.index, name='model'),
    path('api/predict/', views.predict_emotion_api, name='predict_emotion_api'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('contact/', views.contact_view, name='contact'),
    path('privacy/', views.privacy_view, name='privacy'),
    path('terms/', views.terms_view, name='terms'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('insights/', views.insights_view, name='insights'),
    path('profile/', views.profile_view, name='profile'),
    path('account/delete/', views.delete_account_view, name='delete_account'),
    path('history/', views.history_view, name='history'),
    path('history/delete/<int:item_id>/', views.delete_history_item, name='delete_history_item'),
    path('history/delete-all/', views.delete_all_history, name='delete_all_history'),
    path('verify-email/', views.verify_email_view, name='verify_email'),
    path('password-reset/', views.password_reset_view, name='password_reset'),
    path('verify-otp/', views.verify_otp_view, name='verify_otp'),
]
