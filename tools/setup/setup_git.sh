#!/usr/bin/env bash

current_email=$(git config --local user.email)

read -p "Enter Your Name: " username
read -p "Enter Your Email: " useremail
git config --global user.name "${username}"
git config --global user.email "${useremail}"
git config --global core.editor vim

git config --local core.untrackedCache true
git config --local core.hooksPath tools/git_hooks/
git config --local commit.template tools/git_hooks/git_message.txt
git config --global alias.co checkout
git config --global alias.br "branch -v"
git config --global alias.ci commit
git config --global alias.st "status -sb"
git config --global alias.r "remote -v"
git config --global alias.amend "commit --amend"
git config --global alias.fork "push -f --set-upstream origin"
# rebase by default
git config --global pull.rebase true
git config --global alias.last "diff HEAD~1..HEAD"
git config --global alias.cp "cherry-pick"
git config --global alias.dc "diff --cached"
git config --global alias.rbc "rebase --continue"
git config --global alias.rbs "rebase --skip"

git config --global alias.unstage "reset HEAD"
git config --global alias.uncommit "reset --soft HEAD^"
git config --global alias.lg "log --pretty=tformat:'%C(yellow)%h %Cgreen(%ad)%Cred%d %Creset%s %C(bold blue)<%cn>%Creset' --decorate --date=short --date=local --graph --all"
git config --global alias.ll "log --pretty=tformat:'%C(yellow)%h %Cgreen(%ad)%Cred%d %Creset%s %C(bold blue)<%cn>%Creset' --decorate --date=short --date=local --numstat"
