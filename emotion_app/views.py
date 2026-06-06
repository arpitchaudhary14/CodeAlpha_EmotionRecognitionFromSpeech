import os
import json
import urllib.request
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
import random
from django.utils import timezone
from datetime import timedelta
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from django.db.models import Count, Avg
from django.contrib.auth.models import User
from .models import PredictionHistory, UserProfile
from .forms import UserRegisterForm

# Import our ML prediction function
from .ml_model.predict import predict_emotion

def send_custom_email(subject, html_message, recipient_list):
    """
    Sends email via a Google Apps Script Webhook (to bypass Render SMTP blocks),
    or falls back to standard Django SMTP if the webhook URL is not provided.
    """
    webhook_url = os.getenv('GMAIL_WEBHOOK_URL')
    
    if webhook_url:
        data = {
            "to": recipient_list[0],
            "subject": subject,
            "htmlBody": html_message
        }
        req = urllib.request.Request(webhook_url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10)
    else:
        send_mail(
            subject,
            "Please view this email in an HTML compatible client.",
            settings.DEFAULT_FROM_EMAIL,
            recipient_list,
            fail_silently=False,
            html_message=html_message
        )

def landing(request):
    """Render the hero landing page."""
    return render(request, 'emotion_app/landing.html')

def search_view(request):
    query = request.GET.get('q', '').strip()
    results = []
    
    if query:
        # Search User History (if logged in)
        if request.user.is_authenticated:
            history = PredictionHistory.objects.filter(user=request.user, emotion__icontains=query)
            for item in history:
                results.append({
                    'title': f"Speech Analysis: {item.emotion.title()} ({item.confidence}%)",
                    'url': '/history/',
                    'type': 'History Record'
                })
                
        # Search Static Pages
        pages = [
            {'title': 'Home / Landing', 'url': '/', 'keywords': 'home landing index start'},
            {'title': 'Speech Analyzer', 'url': '/model/', 'keywords': 'model analyzer speech audio predict detect voice ai'},
            {'title': 'Dashboard', 'url': '/dashboard/', 'keywords': 'dashboard stats analytics recent overview'},
            {'title': 'Prediction History', 'url': '/history/', 'keywords': 'history past records predictions analyses'},
            {'title': 'Account Settings', 'url': '/profile/', 'keywords': 'profile account settings 2fa security password'},
            {'title': 'Contact Us', 'url': '/contact/', 'keywords': 'contact support email phone help'},
            {'title': 'Privacy Policy', 'url': '/privacy/', 'keywords': 'privacy policy data protection secure'},
            {'title': 'Terms of Service', 'url': '/terms/', 'keywords': 'terms of service rules guidelines'},
        ]
        
        for page in pages:
            if query.lower() in page['title'].lower() or query.lower() in page['keywords']:
                results.append({
                    'title': page['title'],
                    'url': page['url'],
                    'type': 'Page'
                })
                
    return render(request, 'emotion_app/search_results.html', {'query': query, 'results': results})

def contact_view(request):
    """Render the contact us page."""
    return render(request, 'emotion_app/contact.html')

def privacy_view(request):
    """Render the privacy policy page."""
    return render(request, 'emotion_app/privacy.html')

def terms_view(request):
    """Render the terms of service page."""
    return render(request, 'emotion_app/terms.html')

@login_required(login_url='login')
def index(request):
    """Render the main frontend interface. Protected by login."""
    return render(request, 'emotion_app/index.html')

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Redirect newly registered users to login, preventing auto-login.
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'emotion_app/register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    timeout_msg = request.GET.get('timeout') == 'true'
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'emotion_app/login.html', {'form': form, 'timeout_msg': timeout_msg})

def logout_view(request):
    timeout = request.GET.get('timeout') == 'true'
    logout(request)
    if timeout:
        return redirect(f"{reverse('login')}?timeout=true")
    return redirect('landing')

@csrf_exempt  # For development convenience. (In production, use CSRF tokens in JS fetch)
def predict_emotion_api(request):
    """
    API endpoint that receives an audio file, saves it temporarily, 
    runs the ML model prediction, and returns the JSON result.
    """
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        
        # Save the uploaded file temporarily using Django's default storage (into MEDIA_ROOT)
        file_name = default_storage.save(audio_file.name, audio_file)
        file_path = default_storage.path(file_name)
        
        try:
            # Run the prediction using our PyTorch backend
            result = predict_emotion(file_path)
            
            if "error" in result:
                return JsonResponse({'success': False, 'message': result['error']}, status=500)
                
            if request.user.is_authenticated:
                PredictionHistory.objects.create(
                    user=request.user,
                    emotion=result['emotion'],
                    confidence=result['confidence']
                )

            return JsonResponse({
                'success': True,
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'probabilities': result.get('probabilities', {})
            })
            
        except Exception as e:
            import traceback
            import logging
            # Log the full error traceback for debugging on Render
            logging.error(f"Prediction Crash: {traceback.format_exc()}")
            # Return a safe JSON response rather than a Django HTML error page
            return JsonResponse({'success': False, 'message': 'Internal Server Error during prediction.', 'details': str(e)}, status=500)
        finally:
            # Clean up: Guaranteed deletion of temporary audio file to save disk space
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    import logging
                    logging.warning(f"Failed to delete temp file {file_path}: {cleanup_error}")
            
    return JsonResponse({'success': False, 'message': 'No audio file provided or invalid request method (must be POST).'}, status=400)

@login_required(login_url='login')
def dashboard_view(request):
    user_history = PredictionHistory.objects.filter(user=request.user)
    
    total_analyses = user_history.count()
    recent_analyses = user_history.order_by('-created_at')[:5]
    
    context = {
        'total_analyses': total_analyses,
        'recent_analyses': recent_analyses,
    }
    return render(request, 'emotion_app/dashboard.html', context)

@login_required(login_url='login')
def insights_view(request):
    user_history = PredictionHistory.objects.filter(user=request.user)
    
    # Emotion Distribution for Chart.js
    distribution = user_history.values('emotion').annotate(count=Count('emotion')).order_by('-count')
    emotion_labels = [item['emotion'].title() for item in distribution]
    emotion_data = [item['count'] for item in distribution]
    
    most_detected_emotion = distribution.first()['emotion'] if distribution.exists() else "None"
    
    # Average Confidence
    avg_conf_agg = user_history.aggregate(Avg('confidence'))
    avg_confidence = round(avg_conf_agg['confidence__avg'] * 100) if avg_conf_agg['confidence__avg'] else 0
    
    # Last Analysis Date
    last_analysis = user_history.order_by('-created_at').first()
    last_analysis_date = last_analysis.created_at if last_analysis else None
    
    context = {
        'most_detected_emotion': most_detected_emotion,
        'avg_confidence': avg_confidence,
        'last_analysis_date': last_analysis_date,
        'emotion_labels': emotion_labels,
        'emotion_data': emotion_data,
    }
    return render(request, 'emotion_app/insights.html', context)

@login_required(login_url='login')
def history_view(request):
    analyses = PredictionHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'emotion_app/history.html', {'analyses': analyses})

@login_required(login_url='login')
def profile_view(request):
    if request.method == 'POST':
        # Handle profile updates, e.g. toggling 2FA
        action = request.POST.get('action')
        profile = request.user.userprofile
        
        if action == 'toggle_2fa':
            profile.two_factor_enabled = not profile.two_factor_enabled
            profile.save()
            return redirect('profile')
            
        elif action == 'upload_pic' and request.FILES.get('profile_pic'):
            profile.profile_pic = request.FILES['profile_pic']
            profile.save()
            return redirect('profile')
            
        elif action == 'update_email' and request.POST.get('new_email'):
            request.user.email = request.POST.get('new_email')
            request.user.save()
            # Also reset verification status if they change email
            profile.email_verified = False
            profile.save()
            return redirect('profile')
            
    return render(request, 'emotion_app/profile.html')

@login_required(login_url='login')
def verify_email_view(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        profile = request.user.userprofile
        if action == 'send_otp':
            # Check cooldown
            if profile.otp_cooldown_until and timezone.now() < profile.otp_cooldown_until:
                minutes_left = int((profile.otp_cooldown_until - timezone.now()).total_seconds() / 60)
                return render(request, 'emotion_app/verify_email.html', {'step': 'request', 'error': f'Please wait {minutes_left} minutes before requesting another OTP.'})

            # Check limits
            if profile.otp_request_count >= 3:
                profile.otp_cooldown_until = timezone.now() + timedelta(minutes=30)
                profile.otp_request_count = 0
                profile.save()
                return render(request, 'emotion_app/verify_email.html', {'step': 'request', 'error': 'Too many requests. Please try again after 30 minutes.'})

            otp = str(random.randint(100000, 999999))
            profile.otp_code = otp
            profile.otp_created_at = timezone.now()
            profile.otp_request_count += 1
            profile.save()
            
            html_message = render_to_string('emotion_app/email_template.html', {
                'otp': otp,
                'user': request.user,
            })
            
            try:
                send_custom_email(
                    'Verify your Email',
                    html_message,
                    [request.user.email]
                )
            except Exception as e:
                import logging
                logging.error(f"Email sending failed: {e}")
                logging.warning(f"RENDER FREE TIER BYPASS: The OTP for {request.user.username} is {otp}")
                return render(request, 'emotion_app/verify_email.html', {'step': 'verify', 'error': 'Failed to send email (Render blocks SMTP). Check the server logs for your OTP!'})
            return render(request, 'emotion_app/verify_email.html', {'step': 'verify'})
        elif action == 'verify_otp':
            otp_entered = request.POST.get('otp')
            if profile.otp_code == otp_entered:
                # Check expiration (2 mins)
                time_diff = timezone.now() - profile.otp_created_at
                if time_diff.total_seconds() <= 120:
                    profile.email_verified = True
                    profile.otp_request_count = 0
                    profile.otp_cooldown_until = None
                    profile.save()
                    return redirect('profile')
                else:
                    return render(request, 'emotion_app/verify_email.html', {'step': 'verify', 'error': 'OTP expired (Valid for 2 minutes only)'})
            else:
                return render(request, 'emotion_app/verify_email.html', {'step': 'verify', 'error': 'Invalid OTP'})
                
    return render(request, 'emotion_app/verify_email.html', {'step': 'request'})

def password_reset_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            otp = str(random.randint(100000, 999999))
            profile = user.userprofile
            
            # Check cooldown
            if profile.otp_cooldown_until and timezone.now() < profile.otp_cooldown_until:
                minutes_left = int((profile.otp_cooldown_until - timezone.now()).total_seconds() / 60)
                return render(request, 'emotion_app/password_reset.html', {'error': f'Please wait {minutes_left} minutes before requesting another OTP.'})

            # Check limits
            if profile.otp_request_count >= 3:
                profile.otp_cooldown_until = timezone.now() + timedelta(minutes=30)
                profile.otp_request_count = 0
                profile.save()
                return render(request, 'emotion_app/password_reset.html', {'error': 'Too many requests. Please try again after 30 minutes.'})

            otp = str(random.randint(100000, 999999))
            profile.otp_code = otp
            profile.otp_created_at = timezone.now()
            profile.otp_request_count += 1
            profile.save()
            
            html_message = render_to_string('emotion_app/email_template.html', {
                'otp': otp,
                'user': user,
            })
            
            try:
                send_custom_email(
                    'Password Reset OTP',
                    html_message,
                    [email]
                )
            except Exception as e:
                import logging
                logging.error(f"Email sending failed: {e}")
                logging.warning(f"RENDER FREE TIER BYPASS: The password reset OTP for {email} is {otp}")
                request.session['reset_email'] = email
                return render(request, 'emotion_app/verify_otp.html', {'error': 'Failed to send email (Render blocks SMTP). Check the server logs for your OTP!'})
            # Store email in session to verify OTP later
            request.session['reset_email'] = email
            return redirect('verify_otp')
        except User.DoesNotExist:
            return render(request, 'emotion_app/password_reset.html', {'error': 'No user with this email.'})
    return render(request, 'emotion_app/password_reset.html')

def verify_otp_view(request):
    email = request.session.get('reset_email')
    if not email:
        return redirect('password_reset')
        
    if request.method == 'POST':
        otp_entered = request.POST.get('otp')
        new_password = request.POST.get('new_password')
        try:
            user = User.objects.get(email=email)
            profile = user.userprofile
            if profile.otp_code == otp_entered:
                time_diff = timezone.now() - profile.otp_created_at
                if time_diff.total_seconds() <= 120:
                    user.set_password(new_password)
                    user.save()
                    profile.otp_request_count = 0
                    profile.otp_cooldown_until = None
                    profile.save()
                    del request.session['reset_email']
                    return redirect('login')
                else:
                    return render(request, 'emotion_app/verify_otp.html', {'error': 'OTP expired (Valid for 2 minutes only)'})
            else:
                return render(request, 'emotion_app/verify_otp.html', {'error': 'Invalid OTP'})
        except User.DoesNotExist:
            return redirect('password_reset')
            
    return render(request, 'emotion_app/verify_otp.html')

@login_required(login_url='login')
def delete_history_item(request, item_id):
    if request.method == 'POST':
        try:
            item = PredictionHistory.objects.get(id=item_id, user=request.user)
            item.delete()
        except PredictionHistory.DoesNotExist:
            pass
    return redirect('history')

@login_required(login_url='login')
def delete_all_history(request):
    if request.method == 'POST':
        PredictionHistory.objects.filter(user=request.user).delete()
    return redirect('history')

@login_required(login_url='login')
def delete_account_view(request):
    if request.method == 'POST':
        username_confirm = request.POST.get('username_confirm')
        if username_confirm == request.user.username:
            user = request.user
            logout(request)
            user.delete()
            return redirect('landing')
        else:
            return redirect('profile')
    return redirect('profile')
