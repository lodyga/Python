from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404

from item.models import Item

@login_required
def index_view(request):
    items = Item.objects.filter(created_by=request.user)
    # items = Item.objects.all()
    context = {"items": items}

    return render(request, "dashboard/index_view.html", context)

