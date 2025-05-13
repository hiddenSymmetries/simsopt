# based on http://protips.readthedocs.io/link-roles.html

from docutils import nodes
from subprocess import check_output


def run_cmd_get_output(cmd):
    return check_output(cmd).strip()


def get_github_rev():
    path = run_cmd_get_output(['git', 'rev-parse', '--short', 'HEAD'])
    try:
        tag = run_cmd_get_output(['git', 'describe', '--exact-match'])
        print('Git commit ID: ', path)
        if len(tag):
            print('Git tag: ', tag)
            path = tag
    except:
        continue
    return path


def setup(app):
    baseurl = 'https://github.com/hiddenSymmetries/simsopt'
    rev = get_github_rev()
    app.add_role('simsopt', autolink('{}/tree/{}/%s'.format(baseurl, rev)))
    app.add_role('simsoptpy', autolink('{}/tree/{}/src/simsopt/%s'.format(baseurl, rev)))
    app.add_role('simsoptpy_file', autolink('{}/blob/{}/src/simsopt/%s'.format(baseurl, rev)))
    app.add_role('simsoptpp', autolink('{}/tree/{}/src/simsoptpp/%s'.format(baseurl, rev)))
    app.add_role('simsoptpp_file', autolink('{}/blob/{}/src/simsoptpp/%s'.format(baseurl, rev)))
    app.add_role('example', autolink('{}/tree/{}/examples/%s'.format(baseurl, rev)))
    app.add_role('example_file', autolink('{}/blob/{}/examples/%s'.format(baseurl, rev)))
    app.add_role('tests', autolink('{}/blob/{}/tests/%s'.format(baseurl, rev)))
    app.add_role('tests_file', autolink('{}/blob/{}/tests/%s'.format(baseurl, rev)))


def autolink(pattern):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = pattern % (text,)
        node = nodes.reference(rawtext, text, refuri=url, **options)
        return [node], []
    return role
