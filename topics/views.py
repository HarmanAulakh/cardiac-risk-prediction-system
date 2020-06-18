from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.db.models import Q
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.db import connection


# Create your views here.

def listing(request):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM topics, course, level WHERE topics_level_id = level_id AND course_id = topics_course_id")
    topicslist = dictfetchall(cursor)

    context = {
        "topicslist": topicslist
    }

    # Message according medicines Role #
    context['heading'] = "Topics Details";
    return render(request, 'topics-details.html', context)

def lists(request, id):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM topics, course, level WHERE topics_level_id = level_id AND course_id = topics_course_id AND topics_course_id = " + id)
    topicslist = dictfetchall(cursor)

    context = {
        "topicslist": topicslist
    }

    # Message according medicines Role #
    context['heading'] = "Topics Details";
    return render(request, 'topics-list.html', context)

def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def getData(id):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM topics WHERE topics_id = " + id)
    dataList = dictfetchall(cursor)
    return dataList[0];

def update(request, topicsId):
    context = {
        "fn": "update",
        "topicsDetails": getData(topicsId),
        "levelslist": getDropDown('level', 'level_id'),
        "courseylist": getDropDown('course', 'course_id'),
        "heading": 'Update Topics',
    }
    if (request.method == "POST"):
        cursor = connection.cursor()
        cursor.execute("""
                   UPDATE topics
                   SET topics_name=%s, topics_desc=%s, topics_course_id=%s, topics_level_id=%s WHERE topics_id = %s
                """, (
            request.POST['topics_name'],
            request.POST['topics_desc'],
            request.POST['topics_course_id'],
            request.POST['topics_level_id'],
            topicsId
        ))
        context["topicsDetails"] =  getData(topicsId)
        messages.add_message(request, messages.INFO, "Topics updated succesfully !!!")
        return redirect('topics-listing')
    else:
        return render(request, 'topics.html', context)


def add(request):
    context = {
        "fn": "add",
        "courseylist": getDropDown('course', 'course_id'),
        "levelslist": getDropDown('level', 'level_id'),
        "heading": 'Add Topics'
    };
    if (request.method == "POST"):
        cursor = connection.cursor()
        cursor.execute("""
		   INSERT INTO topics
		   SET topics_name=%s, topics_desc=%s, topics_course_id=%s, topics_level_id=%s
		""", (
            request.POST['topics_name'],
            request.POST['topics_desc'],
            request.POST['topics_course_id'],
            request.POST['topics_level_id']))
        return redirect('topics-listing')
    return render(request, 'topics.html', context)

def delete(request, id):
    cursor = connection.cursor()
    sql = 'DELETE FROM topics WHERE topics_id=' + id
    cursor.execute(sql)
    messages.add_message(request, messages.INFO, "Topics Deleted succesfully !!!")
    return redirect('topics-listing')

def getDropDown(table, condtion):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM " + table + " WHERE " + condtion)
    dropdownList = dictfetchall(cursor)
    return dropdownList;