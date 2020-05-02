FROM node:10.4.1 as react-build

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . . 

# 
RUN yarn
RUN yarn build

FROM nginx:1.12-alpine
COPY --from=react-build /usr/src/app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
